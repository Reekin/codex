//! Serves `thread/toolCleanup/read` and `thread/toolCleanup/apply` for loaded threads.

use codex_app_server_protocol::ClientResponsePayload;
use codex_app_server_protocol::JSONRPCErrorError;
use codex_app_server_protocol::ThreadToolCleanupApplyParams;
use codex_app_server_protocol::ThreadToolCleanupApplyResponse;
use codex_app_server_protocol::ThreadToolCleanupReadParams;
use codex_app_server_protocol::ThreadToolCleanupReadResponse;
use codex_app_server_protocol::ThreadToolCleanupStatus;
use codex_core::ToolCleanupStatus;

use super::ThreadRequestProcessor;
use super::thread_input::ensure_direct_input_allowed;
use crate::error_code::internal_error;

impl ThreadRequestProcessor {
    pub(crate) async fn thread_tool_cleanup_read(
        &self,
        params: ThreadToolCleanupReadParams,
    ) -> Result<Option<ClientResponsePayload>, JSONRPCErrorError> {
        let (_, thread) = self.load_thread(&params.thread_id).await?;
        let status = thread
            .tool_cleanup_status()
            .await
            .map_err(|err| internal_error(format!("failed to read tool cleanup: {err}")))?;
        Ok(Some(
            ThreadToolCleanupReadResponse {
                status: status_payload(status),
            }
            .into(),
        ))
    }

    pub(crate) async fn thread_tool_cleanup_apply(
        &self,
        params: ThreadToolCleanupApplyParams,
    ) -> Result<Option<ClientResponsePayload>, JSONRPCErrorError> {
        let (_, thread) = self.load_thread(&params.thread_id).await?;
        ensure_direct_input_allowed(thread.as_ref()).await?;
        let outcome = thread
            .apply_tool_cleanup()
            .await
            .map_err(|err| internal_error(format!("failed to apply tool cleanup: {err}")))?;
        Ok(Some(
            ThreadToolCleanupApplyResponse {
                released_tokens: tokens(outcome.released_tokens),
                status: status_payload(outcome.status),
            }
            .into(),
        ))
    }
}

fn status_payload(status: ToolCleanupStatus) -> ThreadToolCleanupStatus {
    ThreadToolCleanupStatus {
        marking: status.marking,
        pending_savings_tokens: tokens(status.pending_savings_tokens),
        required_savings_tokens: tokens(status.required_savings_tokens),
    }
}

fn tokens(value: usize) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}
