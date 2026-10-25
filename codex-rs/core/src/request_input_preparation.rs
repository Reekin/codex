//! A disposable model-specific projection. Only owned outbound requests are changed here.
use codex_api::ResponsesApiRequest;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ModelInputLimits;
use codex_utils_cache::sha1_digest;
use codex_utils_image::PromptImageMode;
use codex_utils_image::PromptImageResizeLimits;
use codex_utils_image::load_data_url_for_prompt_uncached;
use serde::Serialize;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;

const OMITTED_IMAGE: &str = "[image omitted to fit request byte limit]";
const MAX_CACHE_BYTES: usize = 64 * 1024 * 1024;
const MAX_CACHE_ENTRIES: usize = 256;
/// Prepared images are re-encoded as JPEG at this quality when that is smaller and nothing is
/// transparent. Providers price images by pixel size, so only request bytes change.
const JPEG_QUALITY: u8 = 85;
/// Crossing the byte budget omits the oldest images until the request is back down to this
/// share of it, so the prefix changes once per several images instead of on every new one.
const REFILL_PERCENT: u64 = 70;
type CacheKey = ([u8; 20], u32);
static IMAGE_CACHE: Mutex<VecDeque<(CacheKey, Arc<str>)>> = Mutex::new(VecDeque::new());

/// Images a session has omitted to fit its byte budget. They stay omitted, so later requests
/// keep the same prefix instead of recomputing which images fit.
#[derive(Debug, Default)]
pub(crate) struct OmittedImages(Mutex<HashSet<[u8; 20]>>);

fn prepared_image(url: &str, dimension: u32) -> Result<Arc<str>> {
    let key = (sha1_digest(url.as_bytes()), dimension);
    let mut cache = IMAGE_CACHE
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if let Some(index) = cache.iter().position(|(cached, _)| cached == &key) {
        let entry = cache.remove(index).expect("cached image exists");
        let result = Arc::clone(&entry.1);
        cache.push_back(entry);
        return Ok(result);
    }
    drop(cache);
    let image = load_data_url_for_prompt_uncached(
        url,
        PromptImageMode::ResizeWithLimits(PromptImageResizeLimits {
            max_dimension: dimension,
            max_patches: usize::MAX,
        }),
    )
    .map_err(|error| {
        CodexErr::InvalidRequest(format!("Request image preparation failed: {error}"))
    })?;
    let result: Arc<str> = match jpeg(&image.bytes, &image.mime) {
        Some(jpeg) => codex_utils_image::data_url_from_bytes("image/jpeg", &jpeg),
        None => image.into_data_url(),
    }
    .into();
    if result.len() <= MAX_CACHE_BYTES {
        let mut cache = IMAGE_CACHE
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some((_, image)) = cache.iter().find(|(cached, _)| cached == &key) {
            return Ok(Arc::clone(image));
        }
        let mut bytes = cache.iter().map(|(_, image)| image.len()).sum::<usize>();
        while cache.len() >= MAX_CACHE_ENTRIES || bytes + result.len() > MAX_CACHE_BYTES {
            if let Some((_, image)) = cache.pop_front() {
                bytes -= image.len();
            }
        }
        cache.push_back((key, Arc::clone(&result)));
    }
    Ok(result)
}

/// A smaller JPEG encoding of an opaque image, if there is one.
fn jpeg(bytes: &[u8], mime: &str) -> Option<Vec<u8>> {
    if mime.eq_ignore_ascii_case("image/jpeg") {
        return None;
    }
    let image = image::load_from_memory(bytes).ok()?;
    if image.color().has_alpha() && image.to_rgba8().pixels().any(|pixel| pixel[3] != u8::MAX) {
        return None;
    }
    let mut encoded = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut encoded, JPEG_QUALITY)
        .encode_image(&image::DynamicImage::ImageRgb8(image.to_rgb8()))
        .ok()?;
    (encoded.len() < bytes.len()).then_some(encoded)
}

pub(crate) fn json_bytes(value: &impl Serialize) -> Result<usize> {
    // Count exactly what serde emits without allocating a second full request buffer.
    struct Counter(usize);
    impl std::io::Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 += bytes.len();
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut counter = Counter(0);
    serde_json::to_writer(&mut counter, value)?;
    Ok(counter.0)
}

/// The adapter measures its complete wire envelope after final metadata is available.
pub(crate) fn prepare(
    request: &mut ResponsesApiRequest,
    limits: Option<ModelInputLimits>,
    omitted: &OmittedImages,
    wire_bytes: impl Fn(&ResponsesApiRequest) -> Result<usize>,
) -> Result<()> {
    let Some(limits) = limits else { return Ok(()) };
    if let Some(dimension) = limits.max_image_dimension {
        visit_images(&mut request.input, |image| {
            if let Image::Generated(result) = image {
                if !result.is_empty() {
                    let prepared = prepared_image(
                        &format!("data:image/png;base64,{result}"),
                        dimension.get(),
                    )?;
                    **result = prepared
                        .split_once(',')
                        .expect("prepared data URL")
                        .1
                        .to_owned();
                }
                return Ok(());
            }
            let url = image.url();
            // External references cannot be inspected locally. Preserve them for the gateway's
            // final validation, while still charging their JSON bytes to the request budget.
            if url
                .get(..5)
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case("data:"))
            {
                *url = prepared_image(url, dimension.get())?.to_string();
            }
            Ok(())
        })?;
    }
    let Some(limit) = limits.max_request_bytes else {
        return Ok(());
    };
    let limit = limit.get();
    let mut omitted = omitted.0.lock().unwrap_or_else(|error| error.into_inner());
    // Images omitted by earlier requests stay omitted, keeping the prefix the provider cached.
    let mut remaining = 0usize;
    visit_images(&mut request.input, |image| {
        if let Some(digest) = image.digest() {
            if omitted.contains(&digest) {
                image.omit();
            } else {
                remaining += 1;
            }
        }
        Ok(())
    })?;
    let mut bytes = wire_bytes(request)? as u64;
    // Replace only image content, retaining every item, ID, text and tool relationship.
    // Chronological traversal makes a fixed input deterministic and retains the newest images;
    // the newest one is omitted only when it alone does not fit.
    if bytes > limit {
        let refill = limit * REFILL_PERCENT / 100;
        visit_images(&mut request.input, |image| {
            let Some(digest) = image.digest() else {
                return Ok(());
            };
            remaining -= 1;
            if bytes > limit || (bytes > refill && remaining > 0) {
                let before = image.bytes()? as u64;
                image.omit();
                bytes = bytes - before + image.bytes()? as u64;
                omitted.insert(digest);
            }
            Ok(())
        })?;
    }
    if bytes > limit {
        return Err(CodexErr::InvalidRequest(format!(
            "Request exceeds max_request_bytes ({limit}): non-image content requires {bytes} bytes; shorten text, instructions or tools"
        )));
    }
    Ok(())
}

enum Image<'a> {
    Message(&'a mut ContentItem),
    Tool(&'a mut FunctionCallOutputContentItem),
    Generated(&'a mut String),
}

impl Image<'_> {
    fn url(&mut self) -> &mut String {
        match self {
            Self::Message(ContentItem::InputImage { image_url, .. })
            | Self::Tool(FunctionCallOutputContentItem::InputImage { image_url, .. }) => image_url,
            _ => unreachable!("image visitor selects only images"),
        }
    }
    fn bytes(&self) -> Result<usize> {
        match self {
            Self::Message(content) => json_bytes(content),
            Self::Tool(content) => json_bytes(content),
            Self::Generated(result) => json_bytes(result),
        }
    }
    /// Identifies a payload that is still present; omitted payloads have none.
    fn digest(&mut self) -> Option<[u8; 20]> {
        match self {
            Self::Message(ContentItem::InputImage { .. })
            | Self::Tool(FunctionCallOutputContentItem::InputImage { .. }) => {
                Some(sha1_digest(self.url().as_bytes()))
            }
            Self::Generated(result) if !result.is_empty() => Some(sha1_digest(result.as_bytes())),
            _ => None,
        }
    }
    fn omit(&mut self) {
        match self {
            Self::Message(content) => {
                **content = ContentItem::InputText {
                    text: OMITTED_IMAGE.into(),
                }
            }
            Self::Tool(content) => {
                **content = FunctionCallOutputContentItem::InputText {
                    text: OMITTED_IMAGE.into(),
                }
            }
            // Keep the call record and required string field. Empty base64 represents an omitted
            // payload without changing its ID, status or revised prompt.
            Self::Generated(result) => result.clear(),
        }
    }
}

fn visit_images(
    input: &mut [ResponseItem],
    mut visit: impl FnMut(&mut Image<'_>) -> Result<()>,
) -> Result<()> {
    for item in input {
        match item {
            ResponseItem::ImageGenerationCall { result, .. } => {
                visit(&mut Image::Generated(result))?;
            }
            ResponseItem::Message { content, .. } => {
                for content in content {
                    if matches!(content, ContentItem::InputImage { .. }) {
                        visit(&mut Image::Message(content))?;
                    }
                }
            }
            ResponseItem::FunctionCallOutput { output, .. }
            | ResponseItem::CustomToolCallOutput { output, .. } => {
                if let Some(content) = output.content_items_mut() {
                    for content in content {
                        if matches!(content, FunctionCallOutputContentItem::InputImage { .. }) {
                            visit(&mut Image::Tool(content))?;
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "request_input_preparation_tests.rs"]
mod tests;
