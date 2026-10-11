use super::*;
use pretty_assertions::assert_eq;
use codex_protocol::openai_models::ModelInputLimits;
use core_test_support::responses;
use serde_json::Value;
use std::num::NonZeroU64;
use wiremock::Mock;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn request_input_generated_image_next_turn_omits_payload_without_losing_call()
-> anyhow::Result<()> {
    let server = responses::start_mock_server().await;
    let mut state = 7u32;
    let pixels = (0..512 * 512 * 3)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 24) as u8
        })
        .collect();
    let image = image::RgbImage::from_raw(512, 512, pixels).unwrap();
    let mut bytes = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image).write_to(&mut bytes, image::ImageFormat::Png)?;
    let url = codex_utils_image::data_url_from_bytes("image/png", bytes.get_ref());
    let result = url.split_once(',').unwrap().1;
    let mock = responses::mount_sse_sequence(
        &server,
        vec![
            responses::sse(vec![
                ev_response_created("resp-1"),
                responses::ev_image_generation_call(
                    "ig_original",
                    "completed",
                    "keep revised prompt",
                    result,
                ),
                ev_completed("resp-1"),
            ]),
            responses::sse(vec![ev_response_created("resp-2"), ev_completed("resp-2")]),
            responses::sse(vec![ev_response_created("resp-3"), ev_completed("resp-3")]),
        ],
    )
    .await;
    let mut builder = test_codex()
        .with_model(MODEL)
        .with_model_info_override(MODEL, |model| {
            model.input_limits =
                Some(serde_json::from_value(json!({"max_request_bytes":200000})).unwrap());
        });
    let test = builder.build_with_auto_env(&server).await?;
    test.submit_text_turn("generate a picture").await?;
    test.submit_text_turn("describe the generated picture")
        .await?;
    core_test_support::submit_thread_settings(
        &test.codex,
        codex_protocol::protocol::ThreadSettingsOverrides {
            model: Some("gpt-5.5".into()),
            ..Default::default()
        },
    )
    .await?;
    test.submit_text_turn("inspect the original picture")
        .await?;
    let requests = mock.requests();
    assert_eq!(requests.len(), 3);
    let bounded = requests[1].body_json();
    assert!(serde_json::to_vec(&bounded)?.len() <= 200000);
    let find_call = |body: &Value| {
        body["input"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["type"] == "image_generation_call")
            .unwrap()
            .clone()
    };
    let omitted = find_call(&bounded);
    let mut restored = find_call(&requests[2].body_json());
    assert_eq!(restored["result"], result);
    restored["result"] = json!("");
    // Model switches can add internal metadata, but the generated call identity and text remain.
    for field in ["id", "status", "revised_prompt", "result"] {
        assert_eq!(omitted[field], restored[field]);
    }
    assert_eq!(omitted["status"], "completed");
    assert_eq!(omitted["revised_prompt"], "keep revised prompt");
    test.codex.shutdown_and_wait().await?;
    Ok(())
}

fn image_url(seed: u32) -> String {
    let image = image::RgbImage::from_fn(128, 64, |x, y| {
        image::Rgb([
            (x.wrapping_mul(73) ^ y ^ seed) as u8,
            (y * 71 + x) as u8,
            (x * y + seed) as u8,
        ])
    });
    let mut bytes = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image)
        .write_to(&mut bytes, image::ImageFormat::Png)
        .unwrap();
    codex_utils_image::data_url_from_bytes("image/png", bytes.get_ref())
}

fn image_item(url: &str) -> ResponseItem {
    serde_json::from_value(json!({"type":"message","role":"user","content":[
        {"type":"input_text","text":"keep this text"},
        {"type":"input_image","image_url":url}
    ]}))
    .unwrap()
}

fn images(body: &Value) -> Vec<String> {
    body["input"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|item| {
            item.get("content")
                .or_else(|| item.get("output"))
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|content| content["image_url"].as_str().map(str::to_owned))
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn request_input_websocket_budget_resets_prefix_then_resumes_delta() {
    let server = start_websocket_server(vec![vec![
        vec![ev_response_created("resp-1"), ev_completed("resp-1")],
        vec![ev_response_created("resp-2"), ev_completed("resp-2")],
        vec![ev_response_created("resp-3"), ev_completed("resp-3")],
    ]])
    .await;
    let mut harness = websocket_harness(&server).await;
    let old = image_url(1);
    let newest = image_url(2);
    let budget = old.len().max(newest.len()) as u64 + 2500;
    harness.model_info.input_limits = Some(ModelInputLimits {
        max_request_bytes: NonZeroU64::new(budget),
        ..Default::default()
    });
    let mut session = harness.client.new_session();
    let first = prompt_with_input(vec![image_item(&old)]);
    let second = prompt_with_input(vec![image_item(&old), image_item(&newest)]);
    let third = prompt_with_input(vec![
        image_item(&old),
        image_item(&newest),
        message_item("next"),
    ]);
    for prompt in [&first, &second, &third] {
        stream_until_complete(&mut session, &harness, prompt).await;
    }
    let connection = server.single_connection();
    assert_eq!(connection.len(), 3);
    let bodies: Vec<_> = connection
        .iter()
        .map(|request| request.body_json())
        .collect();
    assert_eq!(images(&bodies[0]), vec![old.clone()]);
    assert_eq!(images(&bodies[1]), vec![newest]);
    assert_eq!(bodies[1].get("previous_response_id"), None);
    assert_eq!(bodies[2]["previous_response_id"], "resp-2");
    assert_eq!(
        bodies[2]["input"],
        serde_json::to_value(&third.input[2..]).unwrap()
    );
    for body in &bodies {
        assert!(serde_json::to_vec(body).unwrap().len() as u64 <= budget);
        assert_eq!(body["prompt_cache_key"], bodies[0]["prompt_cache_key"]);
        assert!(body["prompt_cache_key"].as_str().is_some());
        assert!(
            body["client_metadata"]["x-codex-turn-metadata"]
                .as_str()
                .is_some()
        );
        assert_eq!(
            body["client_metadata"]["x-codex-turn-metadata"],
            bodies[0]["client_metadata"]["x-codex-turn-metadata"]
        );
    }
    let handshake = server.single_handshake();
    assert_eq!(
        handshake.header("session-id"),
        Some(harness.session_id.to_string())
    );
    assert_eq!(
        handshake.header("thread-id"),
        Some(harness.thread_id.to_string())
    );
    assert_eq!(first.input, vec![image_item(&old)]);
    server.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn request_input_http_retry_dimensions_identity_and_unconfigured_model() {
    let server = responses::start_mock_server().await;
    let attempts = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let count = Arc::clone(&attempts);
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .respond_with(move |_: &wiremock::Request| {
            if count.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                ResponseTemplate::new(503)
            } else {
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(responses::sse(vec![
                        ev_response_created("resp"),
                        ev_completed("resp"),
                    ]))
            }
        })
        .mount(&server)
        .await;
    let mut provider = codex_model_provider_info::create_oss_provider_with_base_url(
        &format!("{}/v1", server.uri()),
        WireApi::Responses,
    );
    provider.supports_websockets = false;
    provider.request_max_retries = Some(1);
    let mut harness = websocket_harness_with_provider_options(provider, false, false, &[]).await;
    harness.model_info.input_limits = Some(
        serde_json::from_value(json!({"max_image_dimension":32,"max_request_bytes":10000}))
            .unwrap(),
    );
    let original = image_url(3);
    let prompt = prompt_with_input(vec![
        serde_json::from_value(json!({"type":"function_call","call_id":"call-1","name":"look","arguments":"{}"})).unwrap(),
        serde_json::from_value(json!({"type":"function_call_output","call_id":"call-1","output":[
            {"type":"input_text","text":"keep tool text"},
            {"type":"input_image","image_url":format!("https://example.com/{}", "a".repeat(15000))}
        ]})).unwrap(),
        image_item(&original),
        image_item("https://example.com/external.png"),
    ]);
    let mut session = harness.client.new_session();
    stream_until_complete(&mut session, &harness, &prompt).await;
    harness.model_info.input_limits = None;
    stream_until_complete(&mut session, &harness, &prompt).await;
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[0].body, requests[1].body);
    let prepared: Value = requests[1].body_json().unwrap();
    let unconfigured: Value = requests[2].body_json().unwrap();
    let prepared_images = images(&prepared);
    let decoded = codex_utils_image::load_data_url_for_prompt_uncached(
        &prepared_images[0],
        codex_utils_image::PromptImageMode::Original,
    )
    .unwrap();
    assert_eq!((decoded.width, decoded.height), (32, 16));
    assert_eq!(prepared_images[1], "https://example.com/external.png");
    assert_eq!(images(&unconfigured)[1], original);
    assert_eq!(prepared["input"][0], unconfigured["input"][0]);
    assert_eq!(prepared["input"][1]["call_id"], "call-1");
    assert_eq!(
        prepared["input"][1]["output"][0],
        unconfigured["input"][1]["output"][0]
    );
    assert_eq!(prepared["input"][1]["output"][1]["type"], "input_text");
    assert_eq!(
        prepared["prompt_cache_key"],
        unconfigured["prompt_cache_key"]
    );
    for header in ["session-id", "thread-id", "x-codex-turn-metadata"] {
        assert!(
            requests[1].headers.get(header).is_some(),
            "missing {header}"
        );
        assert_eq!(
            requests[1].headers.get(header),
            requests[2].headers.get(header)
        );
    }
    assert_eq!(
        requests[1].headers["session-id"].to_str().unwrap(),
        harness.session_id.to_string()
    );
    assert_eq!(
        requests[1].headers["thread-id"].to_str().unwrap(),
        harness.thread_id.to_string()
    );
    assert!(prepared["prompt_cache_key"].as_str().is_some());
    assert_eq!(prompt.input[2], image_item(&original));
    assert!(requests[1].body.len() <= 10000);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn request_input_catalog_resume_model_switch_preserves_original_history() -> anyhow::Result<()>
{
    let server = responses::start_mock_server().await;
    let mock = responses::mount_sse_sequence(
        &server,
        (0..3)
            .map(|index| {
                let id = format!("resp-{index}");
                responses::sse(vec![ev_response_created(&id), ev_completed(&id)])
            })
            .collect(),
    )
    .await;
    let mut catalog = codex_models_manager::bundled_models_response()?;
    let model = catalog
        .models
        .iter_mut()
        .find(|model| model.slug == MODEL)
        .unwrap();
    model.input_limits = Some(serde_json::from_value(json!({"max_image_dimension":32}))?);
    let mut builder = test_codex()
        .with_model(MODEL)
        .with_pre_build_hook(move |home| {
            let path = home.join("models-override.json");
            std::fs::write(&path, serde_json::to_vec(&catalog).unwrap()).unwrap();
            let mut config = toml::Table::new();
            config.insert(
                "model_catalog_json".into(),
                toml::Value::String(path.to_string_lossy().into_owned()),
            );
            std::fs::write(home.join("config.toml"), toml::to_string(&config).unwrap()).unwrap();
        });
    let test = builder.build_with_auto_env(&server).await?;
    assert!(
        test.config
            .model_catalog
            .as_ref()
            .unwrap()
            .models
            .iter()
            .any(|model| model.slug == MODEL && model.input_limits.is_some())
    );
    let original = image_url(4);
    test.codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Image {
            image_url: original.clone(),
            detail: None,
        }]))
        .await?;
    wait_for_event(&test.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;
    let mut resume_builder = test_codex().with_model(MODEL);
    let resumed = resume_builder.restart(&server, &test).await?;
    resumed.submit_text_turn("retain the image").await?;
    core_test_support::submit_thread_settings(
        &resumed.codex,
        codex_protocol::protocol::ThreadSettingsOverrides {
            model: Some("gpt-5.5".into()),
            ..Default::default()
        },
    )
    .await?;
    resumed
        .submit_text_turn("inspect at original resolution")
        .await?;
    let requests = mock.requests();
    assert_eq!(requests.len(), 3);
    let prepared = images(&requests[0].body_json());
    assert_eq!(images(&requests[1].body_json()), prepared);
    assert_eq!(images(&requests[2].body_json()), vec![original.clone()]);
    assert_ne!(prepared, vec![original]);
    for request in &requests {
        assert!(request.header("thread-id").is_some());
        assert_eq!(request.header("thread-id"), requests[0].header("thread-id"));
    }
    resumed.codex.shutdown_and_wait().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn request_input_websocket_http_fallback_keeps_preparation() {
    let server = responses::start_mock_server().await;
    Mock::given(method("GET"))
        .and(path("/v1/responses"))
        .respond_with(ResponseTemplate::new(426))
        .mount(&server)
        .await;
    let mock = responses::mount_sse_once(
        &server,
        responses::sse(vec![ev_response_created("resp"), ev_completed("resp")]),
    )
    .await;
    let mut provider = codex_model_provider_info::create_oss_provider_with_base_url(
        &format!("{}/v1", server.uri()),
        WireApi::Responses,
    );
    provider.supports_websockets = true;
    let mut harness = websocket_harness_with_provider_options(provider, false, false, &[]).await;
    harness.model_info.input_limits = Some(
        serde_json::from_value(json!({"max_image_dimension":32,"max_request_bytes":10000}))
            .unwrap(),
    );
    let prompt = prompt_with_input(vec![image_item(&image_url(5))]);
    stream_until_complete(&mut harness.client.new_session(), &harness, &prompt).await;
    let body = mock.single_request().body_json();
    let decoded = codex_utils_image::load_data_url_for_prompt_uncached(
        &images(&body)[0],
        codex_utils_image::PromptImageMode::Original,
    )
    .unwrap();
    assert_eq!((decoded.width, decoded.height), (32, 16));
    assert!(serde_json::to_vec(&body).unwrap().len() <= 10000);
    assert!(
        server
            .received_requests()
            .await
            .unwrap()
            .iter()
            .any(|request| request.method == "GET")
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn request_input_oversized_non_image_fails_before_http_transmission() {
    let server = responses::start_mock_server().await;
    let mut provider = codex_model_provider_info::create_oss_provider_with_base_url(
        &format!("{}/v1", server.uri()),
        WireApi::Responses,
    );
    provider.supports_websockets = false;
    let mut harness = websocket_harness_with_provider_options(provider, false, false, &[]).await;
    harness.model_info.input_limits = Some(ModelInputLimits {
        max_request_bytes: NonZeroU64::new(100),
        ..Default::default()
    });
    let prompt = prompt_with_input_and_instructions(
        vec![message_item("text remains required")],
        &"instructions ".repeat(200),
    );
    let error = harness
        .client
        .new_session()
        .stream(
            &prompt,
            &harness.model_info,
            &harness.session_telemetry,
            harness.effort.clone(),
            harness.summary,
            None,
            &turn_metadata(&harness, None),
            &InferenceTraceContext::disabled(),
        )
        .await
        .err()
        .expect("local budget error");
    assert!(error.to_string().contains("non-image content requires"));
    assert!(server.received_requests().await.unwrap().is_empty());
}
