//! Subagent identity behavior of the multi-agent handlers, enabled by `Feature::SubagentIdentity`.

use super::*;
use crate::agent::AgentIdentity;
use crate::session::session::Session;
use crate::tools::handlers::multi_agents_v2::wait::WaitAgentResult;

fn enable_subagent_identity(turn: &mut TurnContext, multi_agent_v2: bool) {
    let mut config = (*turn.config).clone();
    config
        .features
        .enable(Feature::SubagentIdentity)
        .expect("test config should allow feature update");
    if multi_agent_v2 {
        config
            .features
            .enable(Feature::MultiAgentV2)
            .expect("test config should allow feature update");
        config.multi_agent_v2.min_wait_timeout_ms = 0;
        config.multi_agent_v2.max_wait_timeout_ms = 0;
        config.multi_agent_v2.default_wait_timeout_ms = 0;
    }
    set_turn_config(turn, config);
}

fn timed_out_wait_result(current_agent_name: &str) -> WaitAgentResult {
    WaitAgentResult {
        message: format!(
            "Wait timed out for your mailbox as `{current_agent_name}`. This wait only observes updates delivered to your own mailbox; it does not wait for `/root`'s child tasks unless you are `/root`."
        ),
        timed_out: true,
    }
}

async fn wait_v2(session: Arc<Session>, turn: Arc<TurnContext>) -> WaitAgentResult {
    let output = timeout(
        Duration::from_secs(/*secs*/ 1),
        WaitAgentHandlerV2::default().handle(invocation(
            session,
            turn,
            "wait_agent",
            function_payload(json!({})),
        )),
    )
    .await
    .expect("zero configured timeout should return promptly")
    .expect("wait_agent should succeed");
    let (content, _) = expect_text_output(output);
    serde_json::from_str(&content).expect("wait_agent result should be json")
}

#[tokio::test]
async fn wait_agent_rejects_current_agent_target() {
    let (session, mut turn) = make_session_and_context().await;
    enable_subagent_identity(&mut turn, /*multi_agent_v2*/ false);
    let current_thread_id = session.thread_id;
    let invocation = invocation(
        Arc::new(session),
        Arc::new(turn),
        "wait_agent",
        function_payload(json!({"targets": [current_thread_id.to_string()]})),
    );
    let Err(err) = WaitAgentHandler::default().handle(invocation).await else {
        panic!("current agent target should be rejected");
    };
    let FunctionCallError::RespondToModel(message) = err else {
        panic!("expected respond-to-model error");
    };
    assert!(message.contains("wait_agent cannot wait for the current agent `/root`"));
    assert!(message.contains("Return your result if your task is complete"));
}

#[tokio::test]
async fn multi_agent_v2_wait_agent_names_pathless_subagent_mailbox() {
    let (session, mut turn) = make_session_and_context().await;
    enable_subagent_identity(&mut turn, /*multi_agent_v2*/ true);
    turn.session_source = SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
        parent_thread_id: ThreadId::new(),
        depth: 1,
        agent_path: None,
        agent_nickname: None,
        agent_role: None,
    });
    let current_thread_id = session.thread_id.to_string();

    let result = wait_v2(Arc::new(session), Arc::new(turn)).await;

    assert_eq!(result, timed_out_wait_result(&current_thread_id));
}

#[tokio::test]
async fn multi_agent_v2_spawned_child_lists_and_waits_as_its_canonical_path() {
    let (mut session, mut turn) = make_session_and_context().await;
    let manager = thread_manager();
    let root = manager
        .start_thread(StartThreadOptions::new((*turn.config).clone()))
        .await
        .expect("root thread should start");
    set_agent_control(&mut session, manager.agent_control());
    session.thread_id = root.thread_id;
    enable_subagent_identity(&mut turn, /*multi_agent_v2*/ true);

    let session = Arc::new(session);
    let turn = Arc::new(turn);
    let spawn_output = SpawnAgentHandlerV2::default()
        .handle(invocation(
            session.clone(),
            turn.clone(),
            "spawn_agent",
            function_payload(json!({
                "message": "inspect this repo",
                "task_name": "worker"
            })),
        ))
        .await
        .expect("spawn_agent should succeed");
    let _ = expect_text_output(spawn_output);
    let agent_id = session
        .services
        .local_agent_runtime
        .resolve_agent_reference(session.thread_id, &turn.session_source, "worker")
        .await
        .expect("worker path should resolve");
    let child_thread = manager
        .get_thread(agent_id)
        .await
        .expect("child thread should exist");
    let child_session = Arc::clone(&child_thread.session);
    let child_turn = child_session.new_default_turn().await;

    // The child's identity names the same path that list_agents reports for it.
    let output = ListAgentsHandlerV2
        .handle(invocation(
            Arc::clone(&child_session),
            Arc::clone(&child_turn),
            "list_agents",
            function_payload(json!({})),
        ))
        .await
        .expect("list_agents should succeed");
    let (content, _) = expect_text_output(output);
    let listed: ListAgentsResult =
        serde_json::from_str(&content).expect("list_agents result should be json");
    assert!(
        listed
            .agents
            .iter()
            .any(|agent| agent.agent_name == "/root/worker"),
        "list_agents should report the child's canonical path, got {content}"
    );

    assert_eq!(
        wait_v2(child_session, child_turn).await,
        timed_out_wait_result("/root/worker")
    );
}

#[tokio::test]
async fn multi_agent_v2_spawn_bounds_task_name_and_agent_path_bytes() {
    let (session, mut turn) = make_session_and_context().await;
    enable_subagent_identity(&mut turn, /*multi_agent_v2*/ true);
    let max_task_name = "a".repeat(AgentIdentity::MAX_AGENT_NAME_BYTES);
    let mut parent_path = AgentPath::root();
    while parent_path.as_str().len() + 1 + max_task_name.len()
        <= AgentIdentity::MAX_AGENT_PATH_BYTES
    {
        parent_path = parent_path.join(&max_task_name).expect("valid agent path");
    }
    turn.session_source = SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
        parent_thread_id: ThreadId::new(),
        depth: 1,
        agent_path: Some(parent_path),
        agent_nickname: None,
        agent_role: None,
    });

    let session = Arc::new(session);
    let turn = Arc::new(turn);
    for (task_name, expected) in [
        (
            "BadName".to_string(),
            "agent_name must use only lowercase letters, digits, and underscores".to_string(),
        ),
        (
            "a".repeat(AgentIdentity::MAX_AGENT_NAME_BYTES + 1),
            format!(
                "agent_name must not exceed {} UTF-8 bytes",
                AgentIdentity::MAX_AGENT_NAME_BYTES
            ),
        ),
        (
            max_task_name,
            format!(
                "agent path must not exceed {} UTF-8 bytes",
                AgentIdentity::MAX_AGENT_PATH_BYTES
            ),
        ),
    ] {
        let invocation = invocation(
            Arc::clone(&session),
            Arc::clone(&turn),
            "spawn_agent",
            function_payload(json!({
                "message": "inspect this repo",
                "task_name": task_name
            })),
        );
        let Err(err) = SpawnAgentHandlerV2::default().handle(invocation).await else {
            panic!("invalid agent name should be rejected");
        };
        assert_eq!(err, FunctionCallError::RespondToModel(expected));
    }
}
