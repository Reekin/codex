use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn request_input_compaction_measures_actual_envelope_and_prepares_generated_images()
-> anyhow::Result<()> {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses/compact"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"output":[]})))
        .mount(&server)
        .await;
    let client = ModelClient::new(
        None,
        AgentIdentityAuthPolicy::JwtOnly,
        ThreadId::new(),
        create_oss_provider_with_base_url(&format!("{}/v1", server.uri()), WireApi::Responses),
        SessionSource::Cli,
        "test".into(),
        None,
        true,
        false,
        false,
        None,
        false,
        None,
        HttpClientFactory::new(OutboundProxyPolicy::ReqwestDefault),
    );
    let mut bytes = std::io::Cursor::new(Vec::new());
    image::DynamicImage::new_rgb8(128, 64).write_to(&mut bytes, image::ImageFormat::Png)?;
    let url = codex_utils_image::data_url_from_bytes("image/png", bytes.get_ref());
    let prompt = Prompt {
        input: serde_json::from_value(json!([
            {"type":"image_generation_call","id":"ig_original","status":"completed","revised_prompt":"preserve prompt","result":url.split_once(',').unwrap().1},
            {"type":"message","role":"user","content":[{"type":"input_text","text":"preserve user text"},{"type":"input_image","image_url":url}]}
        ]))?,
        base_instructions: BaseInstructions {
            text: "count these instructions".repeat(20),
            provenance: None,
        },
        ..Default::default()
    };
    let original = prompt.input.clone();
    let metadata = test_responses_metadata_for_client(
        &client,
        None,
        "window".into(),
        None,
        TestCodexResponsesRequestKind::Turn,
    );
    let mut model = test_model_info();
    model.input_limits = Some(serde_json::from_value(json!({"max_image_dimension":32}))?);
    let send = |model: ModelInfo| {
        let client = &client;
        let prompt = &prompt;
        let metadata = &metadata;
        async move {
            client
                .compact_conversation_history(
                    prompt,
                    &model,
                    None,
                    CompactConversationRequestSettings {
                        effort: None,
                        summary: codex_protocol::config_types::ReasoningSummary::None,
                        service_tier: None,
                    },
                    &test_session_telemetry(),
                    &CompactionTraceContext::disabled(),
                    metadata,
                )
                .await
        }
    };
    send(model.clone()).await?;
    let first = server.received_requests().await.unwrap().remove(0);
    let body: serde_json::Value = first.body_json()?;
    let result = body["input"][0]["result"].as_str().unwrap();
    let resized = codex_utils_image::load_data_url_for_prompt_uncached(
        &format!("data:image/png;base64,{result}"),
        codex_utils_image::PromptImageMode::Original,
    )?;
    assert_eq!((resized.width, resized.height), (32, 16));
    let limit = first.body.len() - result.len();
    model.input_limits.as_mut().unwrap().max_request_bytes =
        std::num::NonZeroU64::new(limit as u64);
    send(model.clone()).await?;
    let second = server.received_requests().await.unwrap().remove(1);
    let mut expected = body.clone();
    expected["input"][0]["result"] = json!("");
    assert_eq!(second.body_json::<serde_json::Value>()?, expected);
    assert_eq!(second.body.len(), limit);
    assert_eq!(prompt.input, original);
    model.input_limits.as_mut().unwrap().max_request_bytes = std::num::NonZeroU64::new(1);
    assert!(
        send(model)
            .await
            .unwrap_err()
            .to_string()
            .contains("non-image content requires")
    );
    assert_eq!(server.received_requests().await.unwrap().len(), 2);
    Ok(())
}
