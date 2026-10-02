use super::*;
use pretty_assertions::assert_eq;
use test_case::test_case;

#[test_case("generations", "xhigh", 200; "generate with extended quality")]
#[test_case("edits", "max", 200; "edit with extended quality")]
#[test_case("generations", "max", 400; "preserve unsupported option failure")]
#[tokio::test]
async fn custom_provider_receives_explicit_image_options(
    operation: &str,
    quality: &str,
    status: u16,
) -> Result<()> {
    let server = responses::start_mock_server().await;
    let endpoint = format!("/v1/images/{operation}");
    let response = if status == 200 {
        json!({"created": 1, "data": [{"b64_json": RESULT}], "quality": quality})
    } else {
        json!({"error": {"message": "unsupported image option", "type": "invalid_request_error"}})
    };
    Mock::given(method("POST"))
        .and(path(endpoint.clone()))
        .respond_with(ResponseTemplate::new(status).set_body_json(response))
        .expect(1)
        .mount(&server)
        .await;
    let mut args = json!({
        "prompt": "paint a blue whale",
        "model": "provider-image-model",
        "size": "1536x864",
        "quality": quality,
    });
    if operation == "edits" {
        args["num_last_images_to_include"] = json!(1);
    }
    let response_mock = responses::mount_sse_sequence(
        &server,
        vec![
            responses::sse(vec![
                responses::ev_response_created("resp-1"),
                responses::ev_function_call_with_namespace(
                    "image-options",
                    "image_gen",
                    "imagegen",
                    &args.to_string(),
                ),
                responses::ev_completed("resp-1"),
            ]),
            responses::sse(vec![
                responses::ev_assistant_message("msg-1", "Done"),
                responses::ev_completed("resp-2"),
            ]),
        ],
    )
    .await;
    let codex_home = TempDir::new()?;
    MockResponsesConfig::new(&server.uri())
        .with_model_provider("custom-images")
        .with_provider_name("Custom Responses")
        .with_provider_config(
            "supports_websockets = false\nrequires_openai_auth = false\nenv_key = \"IMAGEGEN_TEST_API_KEY\"",
        )
        .write(codex_home.path())?;
    let mut mcp = TestAppServer::builder()
        .with_codex_home(codex_home.path())
        .with_env_overrides(&[
            ("OPENAI_API_KEY", None),
            ("IMAGEGEN_TEST_API_KEY", Some("image-options-test-key")),
        ])
        .build_initialized_with_timeout(DEFAULT_READ_TIMEOUT)
        .await?;
    let mut input = vec![V2UserInput::Text {
        text: "Generate or edit the image using the requested options".to_string(),
        text_elements: Vec::new(),
    }];
    if operation == "edits" {
        input.push(V2UserInput::Image {
            url: TINY_PNG_DATA_URL.to_string(),
            detail: None,
        });
    }
    start_turn(&mut mcp, ThreadStartParams::default(), input).await?;
    let completed = timeout(
        DEFAULT_READ_TIMEOUT,
        wait_for_image_generation_completed(&mut mcp),
    )
    .await??;
    timeout(
        DEFAULT_READ_TIMEOUT,
        mcp.read_stream_until_notification_message("turn/completed"),
    )
    .await??;
    let ThreadItem::ImageGeneration(image) = completed.item else {
        panic!("expected image generation item");
    };
    assert_eq!(
        image.status,
        if status == 200 { "completed" } else { "failed" }
    );
    let requests = server
        .received_requests()
        .await
        .context("received requests")?;
    let image_requests: Vec<_> = requests
        .iter()
        .filter(|request| request.url.path() == endpoint)
        .collect();
    assert_eq!(image_requests.len(), 1);
    let mut expected = json!({
        "prompt": "paint a blue whale",
        "model": "provider-image-model",
        "size": "1536x864",
        "quality": quality,
        "background": "auto",
    });
    if operation == "edits" {
        expected["images"] = json!([{"image_url": TINY_PNG_DATA_URL}]);
    }
    assert_eq!(
        image_requests[0].body_json::<serde_json::Value>()?,
        expected
    );
    let model_requests = response_mock.requests();
    let body = model_requests[0].body_json();
    let namespace = body["tools"]
        .as_array()
        .context("tools")?
        .iter()
        .find(|tool| tool["name"] == "image_gen")
        .context("image namespace")?;
    let parameters = &namespace["tools"][0]["parameters"];
    for name in ["model", "size", "quality"] {
        assert!(parameters["properties"].get(name).is_some());
        assert!(
            !parameters["required"]
                .as_array()
                .context("required fields")?
                .contains(&json!(name))
        );
        assert!(parameters["properties"][name].to_string().contains("null"));
    }
    let quality_schema = parameters["properties"]["quality"].to_string();
    for value in ["low", "medium", "high", "xhigh", "max", "auto"] {
        assert!(quality_schema.contains(&format!("\"{value}\"")));
    }
    if status != 200 {
        let requests = response_mock.requests();
        let output = requests[1].function_call_output("image-options");
        assert!(output.to_string().contains("unsupported image option"));
    }
    Ok(())
}
