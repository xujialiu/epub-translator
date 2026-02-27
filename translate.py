"""Translate EPUB files using LLMs with configurable settings.

Usage:
    python translate.py <source_path> [options]
    python translate.py book.epub -l Japanese -j 8
    python translate.py book.epub -c configs/my_config.yaml
    python translate.py book.epub --override llm.model=gpt-4 llm.key=sk-xxx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from epub_translator import LLM, FillFailedEvent, SubmitKind, language, translate

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "default.yaml"

# 支持的语言映射（CLI 名称 -> language 模块常量）
_LANGUAGE_MAP: dict[str, str] = {
    "chinese": language.CHINESE,
    "english": language.ENGLISH,
    "japanese": language.JAPANESE,
    "korean": language.KOREAN,
    "spanish": language.SPANISH,
    "french": language.FRENCH,
    "german": language.GERMAN,
    "portuguese": language.PORTUGUESE,
    "traditional_chinese": language.TRADITIONAL_CHINESE,
    "russian": language.RUSSIAN,
    "italian": language.ITALIAN,
    "arabic": language.ARABIC,
    "hindi": language.HINDI,
    "dutch": language.DUTCH,
    "polish": language.POLISH,
    "turkish": language.TURKISH,
    "vietnamese": language.VIETNAMESE,
    "thai": language.THAI,
    "indonesian": language.INDONESIAN,
    "swedish": language.SWEDISH,
    "danish": language.DANISH,
    "norwegian": language.NORWEGIAN,
    "finnish": language.FINNISH,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate EPUB files using LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python translate.py book.epub\n"
            "  python translate.py book.epub -l Japanese -j 8\n"
            "  python translate.py book.epub -c configs/custom.yaml\n"
            "  python translate.py book.epub --override llm.model=gpt-4 llm.key=sk-xxx\n"
        ),
    )
    parser.add_argument("source_path", type=str, help="Path to the source EPUB file")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to custom config YAML file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Target output EPUB path")
    parser.add_argument("-l", "--language", type=str, default=None, help="Target language (e.g., Chinese, Japanese)")
    parser.add_argument("-j", "--concurrency", type=int, default=None, help="Number of concurrent translation jobs")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="OmegaConf dotlist overrides (e.g., llm.model=gpt-4 llm.timeout=120)",
    )
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> DictConfig:
    """加载并合并配置：default.yaml < 用户配置文件 < CLI 参数 < dotlist 覆盖"""
    # 1. 加载默认配置
    default_cfg = OmegaConf.load(_DEFAULT_CONFIG_PATH)
    assert isinstance(default_cfg, DictConfig)

    # 2. 合并用户自定义配置文件
    if args.config is not None:
        user_cfg_path = Path(args.config)
        if not user_cfg_path.exists():
            print(f"Error: Config file '{user_cfg_path}' does not exist")
            sys.exit(1)
        user_cfg = OmegaConf.load(user_cfg_path)
        default_cfg = OmegaConf.merge(default_cfg, user_cfg)
        assert isinstance(default_cfg, DictConfig)

    # 3. 通过 CLI 参数覆盖
    cli_overrides: dict[str, object] = {}
    cli_overrides["source_path"] = args.source_path
    if args.output is not None:
        cli_overrides["target_path"] = args.output
    if args.language is not None:
        cli_overrides["target_language"] = args.language
    if args.concurrency is not None:
        cli_overrides["concurrency"] = args.concurrency

    if cli_overrides:
        default_cfg = OmegaConf.merge(default_cfg, cli_overrides)
        assert isinstance(default_cfg, DictConfig)

    # 4. 通过 --override dotlist 覆盖
    if args.override:
        dotlist_cfg = OmegaConf.from_dotlist(args.override)
        default_cfg = OmegaConf.merge(default_cfg, dotlist_cfg)
        assert isinstance(default_cfg, DictConfig)

    return default_cfg


def _resolve_language(lang_str: str) -> str:
    """将语言字符串解析为 language 模块常量值，支持大小写不敏感匹配。"""
    # 先尝试精确匹配映射表
    lower = lang_str.lower().replace(" ", "_")
    if lower in _LANGUAGE_MAP:
        return _LANGUAGE_MAP[lower]

    # 再检查是否已经是 language 模块中的完整值（如 "Simplified Chinese"）
    all_languages = {v for v in _LANGUAGE_MAP.values()}
    if lang_str in all_languages:
        return lang_str

    # 尝试部分匹配
    for key, value in _LANGUAGE_MAP.items():
        if lower in key or key in lower:
            return value

    # 最后直接使用原始字符串（translate 函数接受任意字符串）
    return lang_str


def _resolve_submit_kind(submit_str: str) -> SubmitKind:
    """将字符串解析为 SubmitKind 枚举。"""
    name = submit_str.upper().strip()
    try:
        return SubmitKind[name]
    except KeyError:
        valid = ", ".join(e.name for e in SubmitKind)
        print(f"Error: Invalid submit kind '{submit_str}'. Valid options: {valid}")
        sys.exit(1)


def _has_overrides(cfg: DictConfig, section: str) -> bool:
    """检查配置中某个 section 是否有非空覆盖值。"""
    if section not in cfg:
        return False
    section_cfg = cfg[section]
    if not isinstance(section_cfg, DictConfig):
        return False
    return any(v is not None for v in section_cfg.values())


def _build_llm_kwargs(cfg: DictConfig) -> dict[str, object]:
    """从配置中构建 LLM 构造函数参数。"""
    llm_cfg = cfg.llm
    kwargs: dict[str, object] = {
        "key": str(llm_cfg.key),
        "url": str(llm_cfg.url),
        "model": str(llm_cfg.model),
        "token_encoding": str(llm_cfg.token_encoding),
    }
    if llm_cfg.timeout is not None:
        kwargs["timeout"] = float(llm_cfg.timeout)
    if llm_cfg.top_p is not None:
        kwargs["top_p"] = llm_cfg.top_p
    if llm_cfg.temperature is not None:
        kwargs["temperature"] = llm_cfg.temperature
    if llm_cfg.retry_times is not None:
        kwargs["retry_times"] = int(llm_cfg.retry_times)
    if llm_cfg.retry_interval_seconds is not None:
        kwargs["retry_interval_seconds"] = float(llm_cfg.retry_interval_seconds)
    if llm_cfg.extra_body is not None:
        extra = OmegaConf.to_container(llm_cfg.extra_body, resolve=True)
        kwargs["extra_body"] = extra
    return kwargs


def _build_llms(cfg: DictConfig) -> dict[str, LLM]:
    """根据配置构建 LLM 实例。返回 {'llm': ...} 或 {'translation_llm': ..., 'fill_llm': ...}。"""
    base_kwargs = _build_llm_kwargs(cfg)
    use_dual = _has_overrides(cfg, "translation") or _has_overrides(cfg, "fill")

    if not use_dual:
        return {"llm": LLM(**base_kwargs)}  # type: ignore[arg-type]

    # 双 LLM 模式：分别为 translation 和 fill 构建独立的 LLM
    translation_kwargs = dict(base_kwargs)
    if _has_overrides(cfg, "translation"):
        for key, val in cfg.translation.items():
            if val is not None:
                translation_kwargs[key] = val

    fill_kwargs = dict(base_kwargs)
    if _has_overrides(cfg, "fill"):
        for key, val in cfg.fill.items():
            if val is not None:
                fill_kwargs[key] = val

    return {
        "translation_llm": LLM(**translation_kwargs),  # type: ignore[arg-type]
        "fill_llm": LLM(**fill_kwargs),  # type: ignore[arg-type]
    }


def _validate_config(cfg: DictConfig) -> None:
    """验证必填配置项。"""
    source = cfg.get("source_path")
    if not source:
        print("Error: source_path is required")
        sys.exit(1)
    if not Path(str(source)).exists():
        print(f"Error: Source file '{source}' does not exist")
        sys.exit(1)

    llm_cfg = cfg.get("llm")
    if not llm_cfg or not llm_cfg.get("key"):
        print("Error: llm.key is required. Set it in config YAML or via --override llm.key=YOUR_KEY")
        sys.exit(1)
    if not llm_cfg.get("url"):
        print("Error: llm.url is required. Set it in config YAML or via --override llm.url=YOUR_URL")
        sys.exit(1)
    if not llm_cfg.get("model"):
        print("Error: llm.model is required. Set it in config YAML or via --override llm.model=YOUR_MODEL")
        sys.exit(1)


def _resolve_target_path(cfg: DictConfig) -> Path:
    """解析输出路径。未指定时自动生成 <source>_translated.epub。"""
    if cfg.get("target_path"):
        return Path(str(cfg.target_path))

    source = Path(str(cfg.source_path))
    return source.parent / f"{source.stem}_translated{source.suffix}"


def _print_token_stats(llms: dict[str, LLM]) -> None:
    """打印 token 使用统计。"""
    print("\n" + "=" * 50)
    print("Token Usage Statistics")
    print("=" * 50)

    if "llm" in llms:
        llm = llms["llm"]
        print(f"\n  Total tokens:       {llm.total_tokens:,}")
        print(f"  Input tokens:       {llm.input_tokens:,}")
        print(f"  Input cache tokens: {llm.input_cache_tokens:,}")
        print(f"  Output tokens:      {llm.output_tokens:,}")
    else:
        t_llm = llms["translation_llm"]
        f_llm = llms["fill_llm"]

        print("\nTranslation LLM:")
        print(f"  Total tokens:       {t_llm.total_tokens:,}")
        print(f"  Input tokens:       {t_llm.input_tokens:,}")
        print(f"  Input cache tokens: {t_llm.input_cache_tokens:,}")
        print(f"  Output tokens:      {t_llm.output_tokens:,}")

        print("\nFill LLM:")
        print(f"  Total tokens:       {f_llm.total_tokens:,}")
        print(f"  Input tokens:       {f_llm.input_tokens:,}")
        print(f"  Input cache tokens: {f_llm.input_cache_tokens:,}")
        print(f"  Output tokens:      {f_llm.output_tokens:,}")

        total_combined = t_llm.total_tokens + f_llm.total_tokens
        input_combined = t_llm.input_tokens + f_llm.input_tokens
        input_cache_combined = t_llm.input_cache_tokens + f_llm.input_cache_tokens
        output_combined = t_llm.output_tokens + f_llm.output_tokens

        print("\nCombined Total:")
        print(f"  Total tokens:       {total_combined:,}")
        print(f"  Input tokens:       {input_combined:,}")
        print(f"  Input cache tokens: {input_cache_combined:,}")
        print(f"  Output tokens:      {output_combined:,}")

    print("=" * 50 + "\n")


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args)
    _validate_config(cfg)

    # 解析配置值
    target_language = _resolve_language(str(cfg.target_language))
    submit_kind = _resolve_submit_kind(str(cfg.submit))
    target_path = _resolve_target_path(cfg)
    concurrency = int(cfg.concurrency)
    max_retries = int(cfg.max_retries)
    max_group_tokens = int(cfg.max_group_tokens)
    user_prompt = str(cfg.user_prompt) if cfg.get("user_prompt") else None

    # 构建 LLM 实例
    llms = _build_llms(cfg)

    print(f"Source:      {cfg.source_path}")
    print(f"Target:      {target_path}")
    print(f"Language:    {target_language}")
    print(f"Submit mode: {submit_kind.name}")
    print(f"Concurrency: {concurrency}")
    if "llm" in llms:
        print(f"LLM:         {cfg.llm.model}")
    else:
        print(f"LLM:         {cfg.llm.model} (dual mode)")
    print()

    # FillFailedEvent 回调
    def on_fill_failed(event: FillFailedEvent) -> None:
        print(f"Retry {event.retried_count} Validation failed:")
        print(f"{event.error_message}")
        print("---\n")
        if event.over_maximum_retries:
            print(
                "+ ===============================\n"
                "  Warning: Maximum retries reached without successful XML filling. Will ignore remaining errors.\n"
                "+ ===============================\n"
            )

    # 进度条
    with tqdm(
        total=100,
        desc="Translating",
        unit="%",
        bar_format="{l_bar}{bar}| {n:.1f}/{total:.0f}% [{elapsed}<{remaining}]",
    ) as pbar:
        last_progress = 0.0

        def on_progress(progress: float) -> None:
            nonlocal last_progress
            increment = (progress - last_progress) * 100
            pbar.update(increment)
            last_progress = progress

        translate(
            source_path=str(cfg.source_path),
            target_path=str(target_path),
            target_language=target_language,
            submit=submit_kind,
            concurrency=concurrency,
            max_retries=max_retries,
            max_group_tokens=max_group_tokens,
            user_prompt=user_prompt,
            on_progress=on_progress,
            on_fill_failed=on_fill_failed,
            **llms,  # type: ignore[arg-type]
        )

    _print_token_stats(llms)
    print(f"Translation complete: {target_path}")


if __name__ == "__main__":
    main()
