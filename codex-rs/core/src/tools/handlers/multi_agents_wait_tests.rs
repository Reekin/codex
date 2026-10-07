use super::*;
use crate::session::TurnInput;
use crate::session::session::Session;
use crate::state::TaskKind;
use crate::tasks::SessionTask;
use crate::tasks::SessionTaskResult;
use futures::FutureExt;
use test_case::test_case;

struct HeldTask;

impl SessionTask for HeldTask {
    fn kind(&self) -> TaskKind {
        TaskKind::Regular
    }

    fn span_name(&self) -> &'static str {
        "session_task.wait_steer_test"
    }

    async fn run(
        self: Arc<Self>,
        _session: Arc<Session>,
        _turn: Arc<TurnContext>,
        _input: Vec<TurnInput>,
        cancellation_token: CancellationToken,
    ) -> SessionTaskResult {
        cancellation_token.cancelled().await;
        Ok(None)
    }
}

#[derive(Clone, Copy)]
enum SteerTiming {
    BeforeWait,
    DuringWait,
    FollowedByMailbox,
}

async fn enqueue_unrelated_mail(session: &Session) {
    session
        .input_queue
        .enqueue_mailbox_communication(
            InterAgentCommunication::new(
                AgentPath::root(),
                AgentPath::root(),
                Vec::new(),
                "unrelated update".to_string(),
                /*trigger_turn*/ false,
            ),
            Default::default(),
        )
        .await;
}

#[test_case(SteerTiming::BeforeWait; "already pending")]
#[test_case(SteerTiming::DuringWait; "during wait")]
#[test_case(SteerTiming::FollowedByMailbox; "coalesced with mailbox")]
#[tokio::test]
async fn wait_agent_returns_on_steer_without_stopping_child(timing: SteerTiming) {
    let (mut session, turn) = make_session_and_context().await;
    let manager = thread_manager();
    session.services.agent_control = manager.agent_control();
    let child = manager
        .start_thread(StartThreadOptions::new((*turn.config).clone()))
        .await
        .expect("start child");
    let child_turn = child
        .thread
        .session
        .new_turn_with_default_settings("held-child".to_string(), Default::default())
        .await;
    child
        .thread
        .session
        .spawn_task(child_turn, Vec::new(), HeldTask)
        .await;
    let child_status = manager.agent_control().get_status(child.thread_id).await;
    assert_eq!(child_status, AgentStatus::Running);
    let session = Arc::new(session);
    let turn = Arc::new(turn);
    session.spawn_task(turn.clone(), Vec::new(), HeldTask).await;
    let turn_state = session
        .input_queue
        .turn_state_for_sub_id(&session.active_turn, &turn.sub_id)
        .await
        .expect("parent turn should be active");

    if matches!(timing, SteerTiming::BeforeWait) {
        session
            .route_realtime_text_input("change direction".to_string())
            .await;
    }

    let handler = WaitAgentHandler::default();
    let wait = handler.handle(invocation(
        session.clone(),
        turn.clone(),
        "wait_agent",
        function_payload(json!({
            "targets": [child.thread_id.to_string()],
            "timeout_ms": MAX_WAIT_TIMEOUT_MS
        })),
    ));
    tokio::pin!(wait);
    if !matches!(timing, SteerTiming::BeforeWait) {
        assert!(wait.as_mut().now_or_never().is_none());
        session
            .route_realtime_text_input("change direction".to_string())
            .await;
        if matches!(timing, SteerTiming::FollowedByMailbox) {
            // Do not poll the waiter between these sends: watch retains only the
            // latest activity, so the queue must preserve the steer wakeup.
            enqueue_unrelated_mail(&session).await;
        }
    }

    let output = timeout(Duration::from_secs(1), wait)
        .await
        .expect("steering should wake the long wait")
        .expect("wait should return valid output");
    let (content, success) = expect_text_output(output);
    assert_eq!(
        serde_json::from_str::<wait::WaitAgentResult>(&content).expect("wait result JSON"),
        wait::WaitAgentResult {
            status: HashMap::new(),
            timed_out: false,
        }
    );
    assert_eq!(success, None);
    assert_eq!(
        manager.agent_control().get_status(child.thread_id).await,
        child_status
    );
    let pending = session
        .input_queue
        .take_pending_input_for_turn_state(turn_state.as_ref())
        .await;
    assert!(matches!(
        pending.as_slice(),
        [TurnInput::UserInput { content, .. }]
            if content == &vec![UserInput::Text {
                text: "change direction".to_string(),
                text_elements: Vec::new(),
            }]
    ));

    session.abort_all_tasks(TurnAbortReason::Interrupted).await;
    child
        .thread
        .submit(Op::Shutdown {})
        .await
        .expect("shutdown");
}

#[tokio::test]
async fn wait_agent_ignores_unrelated_mail_and_returns_child_completion() {
    let (mut session, turn) = make_session_and_context().await;
    let manager = thread_manager();
    session.services.agent_control = manager.agent_control();
    let child = manager
        .start_thread(StartThreadOptions::new((*turn.config).clone()))
        .await
        .expect("start child");
    let session = Arc::new(session);
    let turn = Arc::new(turn);
    enqueue_unrelated_mail(&session).await;
    let handler = WaitAgentHandler::default();
    let wait = handler.handle(invocation(
        session.clone(),
        turn,
        "wait_agent",
        function_payload(json!({
            "targets": [child.thread_id.to_string()],
            "timeout_ms": MAX_WAIT_TIMEOUT_MS
        })),
    ));
    tokio::pin!(wait);
    assert!(wait.as_mut().now_or_never().is_none());
    enqueue_unrelated_mail(&session).await;
    assert!(
        timeout(Duration::from_millis(20), wait.as_mut())
            .await
            .is_err(),
        "queued and newly delivered unrelated mail must not finish a status wait"
    );
    child
        .thread
        .submit(Op::Shutdown {})
        .await
        .expect("shutdown");
    let output = timeout(Duration::from_secs(1), wait)
        .await
        .expect("child completion should wake wait")
        .expect("wait should succeed");
    let (content, _) = expect_text_output(output);
    assert_eq!(
        serde_json::from_str::<wait::WaitAgentResult>(&content).expect("wait result JSON"),
        wait::WaitAgentResult {
            status: HashMap::from([(child.thread_id.to_string(), AgentStatus::Shutdown)]),
            timed_out: false,
        }
    );
    assert!(session.input_queue.has_pending_mailbox_items().await);
}
