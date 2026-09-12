use super::*;
use test_case::test_case;

#[test_case("api"; "API key exposes installed image tool")]
#[test_case("plus"; "ChatGPT retains installed image tool")]
#[test_case("free"; "free plan retains entitlement restriction")]
#[tokio::test]
async fn installed_image_generation_exposure_follows_capabilities_and_entitlement(
    auth: &str,
) -> Result<()> {
    let server = responses::start_mock_server().await;
    let response_mock = responses::mount_sse_once(
        &server,
        responses::sse(vec![
            responses::ev_assistant_message("msg-1", "Done"),
            responses::ev_completed("resp-1"),
        ]),
    )
    .await;
    let codex_home = TempDir::new()?;
    create_config_toml(codex_home.path(), &server.uri(), ImagegenTestMode::Direct)?;
    let api_key = if auth == "api" {
        Some("image-tool-test-key")
    } else {
        write_chatgpt_auth(
            codex_home.path(),
            ChatGptAuthFixture::new("image-tool-test-token").plan_type(auth),
            AuthCredentialsStoreMode::File,
        )?;
        None
    };
    let mut mcp = TestAppServer::builder()
        .with_codex_home(codex_home.path())
        .with_env_overrides(&[("OPENAI_API_KEY", api_key)])
        .build_initialized_with_timeout(DEFAULT_READ_TIMEOUT)
        .await?;
    start_image_generation_turn(&mut mcp, ThreadStartParams::default()).await?;
    timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_notification_message("turn/completed"),
    )
    .await??;

    let request = response_mock.single_request();
    let body = request.body_json();
    let image_tools = body["tools"]
        .as_array()
        .context("model request should advertise tools")?
        .iter()
        .filter(|tool| tool["name"] == "image_gen")
        .flat_map(|namespace| namespace["tools"].as_array().into_iter().flatten())
        .map(|tool| (&tool["type"], &tool["name"]))
        .collect::<Vec<_>>();
    let function_type = json!("function");
    let function_name = json!("imagegen");
    let expected = if auth == "free" {
        vec![]
    } else {
        vec![(&function_type, &function_name)]
    };
    assert_eq!(image_tools, expected);
    assert_eq!(
        request.header("authorization"),
        Some(format!(
            "Bearer {}",
            api_key.unwrap_or("image-tool-test-token")
        ))
    );
    Ok(())
}
