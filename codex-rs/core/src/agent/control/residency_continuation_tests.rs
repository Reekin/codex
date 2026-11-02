use super::*;
use pretty_assertions::assert_eq;

fn child_source(parent_thread_id: ThreadId) -> SessionSource {
    SessionSource::SubAgent(SubAgentSource::ThreadSpawn {
        parent_thread_id,
        depth: 1,
        agent_path: None,
        agent_nickname: None,
        agent_role: None,
    })
}

#[test]
fn child_turns_retain_parent_until_each_result_is_reported() {
    let control = LocalAgentControl::default();
    let parent = ThreadId::new();
    assert!(
        control
            .runtime
            .completion_parent_guard(&SessionSource::Exec)
            .is_none()
    );
    let first = control
        .runtime
        .completion_parent_guard(&child_source(parent))
        .expect("first child turn");
    let followup = control
        .runtime
        .completion_parent_guard(&child_source(parent))
        .expect("second child turn");
    assert!(control.runtime.has_outstanding_child_completion(parent));
    drop(first);
    assert!(control.runtime.has_outstanding_child_completion(parent));
    drop(followup);
    assert!(!control.runtime.has_outstanding_child_completion(parent));
}

#[tokio::test]
async fn parent_flag_decides_child_result_continuation() {
    let mut config = test_config().await;
    let _ = config.features.enable(Feature::MultiAgentV2);
    let temp_home = tempfile::tempdir().expect("create temp home");
    config.codex_home = temp_home.path().to_path_buf().try_into().unwrap();
    config.cwd = temp_home.path().to_path_buf().try_into().unwrap();
    let mut enabled_config = config.clone();
    let _ = enabled_config
        .features
        .enable(Feature::SubagentContinuation);
    let manager = ThreadManager::with_models_provider_and_home_for_tests(
        CodexAuth::from_api_key("dummy"),
        config.model_provider.clone(),
        config.codex_home.to_path_buf(),
        Arc::new(codex_exec_server::EnvironmentManager::default_for_tests()),
    );
    let root = manager
        .start_thread(StartThreadOptions::new(enabled_config.clone()))
        .await
        .expect("start root thread");
    let control = manager.agent_control();
    let state = control
        .runtime
        .upgrade()
        .expect("thread manager should be live");
    let disabled_worker =
        spawn_v2_subagent(&control, &state, config, root.thread_id, "disabled").await;
    // Each parent's own flag counts, whatever its child runs with.
    spawn_v2_subagent(
        &control,
        &state,
        enabled_config,
        disabled_worker.thread_id,
        "enabled",
    )
    .await;
    assert!(
        control
            .runtime
            .parent_continues_subagent_results(root.thread_id)
            .await
    );
    assert!(
        !control
            .runtime
            .parent_continues_subagent_results(disabled_worker.thread_id)
            .await
    );
    assert!(
        !control
            .runtime
            .parent_continues_subagent_results(ThreadId::new())
            .await
    );
}

#[tokio::test]
async fn residency_eviction_waits_for_outstanding_child_results() {
    let mut config = test_config().await;
    let _ = config.features.enable(Feature::MultiAgentV2);
    let _ = config.features.enable(Feature::SubagentContinuation);
    let temp_home = tempfile::tempdir().expect("create temp home");
    config.codex_home = temp_home.path().to_path_buf().try_into().unwrap();
    config.cwd = temp_home.path().to_path_buf().try_into().unwrap();
    let manager = ThreadManager::with_models_provider_and_home_for_tests(
        CodexAuth::from_api_key("dummy"),
        config.model_provider.clone(),
        config.codex_home.to_path_buf(),
        Arc::new(codex_exec_server::EnvironmentManager::default_for_tests()),
    );
    let root = manager
        .start_thread(StartThreadOptions::new(config.clone()))
        .await
        .expect("start root thread");
    let control = manager.agent_control();
    let state = control
        .runtime
        .upgrade()
        .expect("thread manager should be live");
    let membership = control.runtime.admit_start().expect("admit residency work");
    let slot = control
        .reserve_v2_residency_slot(
            &state,
            &config,
            &membership,
            /*protected_thread_id*/ None,
        )
        .await
        .expect("resident slot");
    let worker = spawn_v2_subagent(&control, &state, config, root.thread_id, "worker").await;
    slot.commit(worker.thread_id);
    mark_thread_completed(worker.thread.as_ref()).await;

    // The idle worker still owns a running grandchild whose result it must receive.
    let grandchild_turn = control
        .runtime
        .completion_parent_guard(&child_source(worker.thread_id))
        .expect("grandchild turn");
    assert_eq!(
        manager
            .try_evict_v2_thread(Arc::clone(&worker.thread))
            .await
            .expect("eviction attempt"),
        super::super::ThreadEvictionOutcome::Busy
    );
    drop(grandchild_turn);
    assert_eq!(
        manager
            .try_evict_v2_thread(Arc::clone(&worker.thread))
            .await
            .expect("eviction attempt"),
        super::super::ThreadEvictionOutcome::Evicted
    );
}
