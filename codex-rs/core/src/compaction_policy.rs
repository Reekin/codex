use codex_model_provider::RemoteCompactionSupport;
use codex_protocol::openai_models::ModelInfo;

pub(crate) fn remote_compaction_support(
    provider_support: RemoteCompactionSupport,
    model_info: &ModelInfo,
) -> RemoteCompactionSupport {
    if model_info.supports_remote_compaction {
        provider_support
    } else {
        RemoteCompactionSupport::Unsupported
    }
}
