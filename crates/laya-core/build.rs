//! The `mlx` feature links a JIT build of MLX and ships the matching
//! residual `metal/mlx.metallib`. The vendored `mlx-sys` build script reads
//! `MLX_RS_METAL_JIT` and `MACOSX_DEPLOYMENT_TARGET`; refuse to build an
//! inconsistent combination instead of failing at first Metal use.

fn main() {
    println!("cargo:rerun-if-env-changed=MLX_RS_METAL_JIT");
    println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");
    println!("cargo:rerun-if-changed=metal/mlx.metallib");
    if std::env::var("CARGO_FEATURE_MLX").is_err() {
        return;
    }
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        panic!("laya-core feature `mlx` is only supported on macOS (Apple Silicon)");
    }
    if std::env::var("MLX_RS_METAL_JIT").as_deref() != Ok("1") {
        panic!(
            "laya-core feature `mlx` requires MLX_RS_METAL_JIT=1 so the linked MLX matches the embedded residual metallib; set it in .cargo/config.toml [env] (see laya-rs .cargo/config.toml)"
        );
    }
    match std::env::var("MACOSX_DEPLOYMENT_TARGET") {
        Ok(target) if target == "14.0" => {}
        other => panic!(
            "laya-core feature `mlx` requires MACOSX_DEPLOYMENT_TARGET=14.0 (got {other:?}); the embedded metallib targets macOS 14"
        ),
    }
}
