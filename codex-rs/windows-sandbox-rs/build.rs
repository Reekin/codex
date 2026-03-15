fn main() {
    println!("cargo:rerun-if-changed=codex-windows-sandbox-setup.manifest");

    let mut res = winres::WindowsResource::new();
    res.set_manifest_file("codex-windows-sandbox-setup.manifest");
    res.compile()
        .expect("failed to compile windows sandbox resources");
}
