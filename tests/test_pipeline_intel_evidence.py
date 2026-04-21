# -*- coding: utf-8 -*-
"""
Tests for pipeline-side evidence backfill on qualitative intelligence fields.
"""

import importlib
import os
import sys
import unittest
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@dataclass
class DummySearchResult:
    title: str
    snippet: str
    url: str
    source: str
    published_date: str | None = None


@dataclass
class DummySearchResponse:
    query: str
    results: list
    provider: str
    success: bool = True
    error_message: str | None = None
    search_time: float = 0.0


def _response(provider: str, results) -> DummySearchResponse:
    return DummySearchResponse(
        query="test",
        results=results,
        provider=provider,
        success=True,
    )


class PipelineIntelEvidenceTestCase(unittest.TestCase):
    """Tests for deterministic evidence/citation backfill."""

    def _load_pipeline_types(self):
        analyzer_stub = ModuleType("src.analyzer")

        @dataclass
        class AnalysisResult:
            code: str
            name: str
            sentiment_score: int
            trend_prediction: str
            operation_advice: str
            decision_type: str = "hold"
            confidence_level: str = "中"
            report_language: str = "zh"
            dashboard: dict | None = None
            trend_analysis: str = ""
            short_term_outlook: str = ""
            medium_term_outlook: str = ""
            technical_analysis: str = ""
            ma_analysis: str = ""
            volume_analysis: str = ""
            pattern_analysis: str = ""
            fundamental_analysis: str = ""
            sector_position: str = ""
            company_highlights: str = ""
            news_summary: str = ""
            market_sentiment: str = ""
            hot_topics: str = ""
            analysis_summary: str = ""
            key_points: str = ""
            risk_warning: str = ""
            buy_reason: str = ""
            market_snapshot: dict | None = None
            raw_response: str | None = None
            search_performed: bool = False
            data_sources: str = ""
            success: bool = True
            error_message: str | None = None
            current_price: float | None = None
            change_pct: float | None = None
            model_used: str | None = None
            query_id: str | None = None

        analyzer_stub.AnalysisResult = AnalysisResult
        analyzer_stub.GeminiAnalyzer = object
        analyzer_stub.fill_chip_structure_if_needed = lambda *args, **kwargs: None
        analyzer_stub.fill_price_position_if_needed = lambda *args, **kwargs: None

        config_stub = ModuleType("src.config")

        class Config:
            pass

        config_stub.Config = Config
        config_stub.get_config = lambda: SimpleNamespace(report_language="zh")

        storage_stub = ModuleType("src.storage")
        storage_stub.get_db = lambda: None

        data_provider_stub = ModuleType("data_provider")
        data_provider_stub.DataFetcherManager = object

        data_provider_base_stub = ModuleType("data_provider.base")
        data_provider_base_stub.normalize_stock_code = lambda code: code

        data_provider_rt_stub = ModuleType("data_provider.realtime_types")
        data_provider_rt_stub.ChipDistribution = object

        data_provider_us_stub = ModuleType("data_provider.us_index_mapping")
        data_provider_us_stub.is_us_stock_code = lambda code: False

        stock_mapping_stub = ModuleType("src.data.stock_mapping")
        stock_mapping_stub.STOCK_NAME_MAP = {}

        notification_stub = ModuleType("src.notification")
        notification_stub.NotificationService = object
        notification_stub.NotificationChannel = object

        report_language_stub = ModuleType("src.report_language")
        report_language_stub.get_unknown_text = lambda language="zh": "未知"
        report_language_stub.localize_confidence_level = lambda level, language="zh": level
        report_language_stub.normalize_report_language = lambda language="zh": (language or "zh")

        search_stub = ModuleType("src.search_service")

        class SearchService:
            @staticmethod
            def is_index_or_etf(code: str, name: str) -> bool:
                return False

        search_stub.SearchService = SearchService

        social_sentiment_stub = ModuleType("src.services.social_sentiment_service")
        social_sentiment_stub.SocialSentimentService = object

        enums_stub = ModuleType("src.enums")

        class ReportType:
            SIMPLE = SimpleNamespace(value="simple")

        enums_stub.ReportType = ReportType

        stock_analyzer_stub = ModuleType("src.stock_analyzer")
        stock_analyzer_stub.StockTrendAnalyzer = object
        stock_analyzer_stub.TrendAnalysisResult = object

        trading_calendar_stub = ModuleType("src.core.trading_calendar")
        trading_calendar_stub.get_effective_trading_date = lambda *args, **kwargs: None
        trading_calendar_stub.get_market_for_stock = lambda code: "cn"
        trading_calendar_stub.get_market_now = lambda market: SimpleNamespace(date=lambda: None)
        trading_calendar_stub.is_market_open = lambda *args, **kwargs: False

        bot_models_stub = ModuleType("bot.models")
        bot_models_stub.BotMessage = object

        with patch.dict(
            sys.modules,
            {
                "src.analyzer": analyzer_stub,
                "src.config": config_stub,
                "src.storage": storage_stub,
                "data_provider": data_provider_stub,
                "data_provider.base": data_provider_base_stub,
                "data_provider.realtime_types": data_provider_rt_stub,
                "data_provider.us_index_mapping": data_provider_us_stub,
                "src.data.stock_mapping": stock_mapping_stub,
                "src.notification": notification_stub,
                "src.report_language": report_language_stub,
                "src.search_service": search_stub,
                "src.services.social_sentiment_service": social_sentiment_stub,
                "src.enums": enums_stub,
                "src.stock_analyzer": stock_analyzer_stub,
                "src.core.trading_calendar": trading_calendar_stub,
                "bot.models": bot_models_stub,
            },
            clear=False,
        ):
            pipeline_module = importlib.import_module("src.core.pipeline")

        pipeline = pipeline_module.StockAnalysisPipeline.__new__(pipeline_module.StockAnalysisPipeline)
        pipeline.config = SimpleNamespace(report_language="zh")
        core_package = sys.modules.get("src.core")
        if core_package is not None and getattr(core_package, "pipeline", None) is pipeline_module:
            delattr(core_package, "pipeline")
        sys.modules.pop("src.core.pipeline", None)
        return pipeline, AnalysisResult

    def test_backfill_grounds_missing_intelligence_fields(self) -> None:
        """Missing qualitative fields should be rewritten with explicit citations."""
        pipeline, AnalysisResult = self._load_pipeline_types()
        result = AnalysisResult(
            code="600519",
            name="贵州茅台",
            sentiment_score=60,
            trend_prediction="震荡",
            operation_advice="观望",
            analysis_summary="市场情绪中性",
            dashboard={"intelligence": {}, "core_conclusion": {"one_sentence": "观望等待确认"}},
            data_sources="技术面数据",
        )

        intel_results = {
            "latest_news": _response(
                "Bocha",
                [
                    DummySearchResult(
                        title="公司发布年度分红方案",
                        snippet="董事会审议通过现金分红方案。",
                        url="https://example.com/dividend",
                        source="财联社",
                        published_date="2026-04-20",
                    )
                ],
            ),
            "market_analysis": _response(
                "Tavily",
                [
                    DummySearchResult(
                        title="机构维持买入评级",
                        snippet="研报认为利润率和现金流维持稳健。",
                        url="https://example.com/rating",
                        source="证券时报",
                        published_date="2026-04-19",
                    )
                ],
            ),
        }
        fundamental_context = {
            "earnings": {
                "data": {
                    "forecast_summary": "业绩预告显示净利润同比增长约20%。",
                }
            },
            "source_chain": [
                "earnings_forecast:akshare_profit_forecast_em",
                "earnings_quick:akshare_report_em",
            ],
        }

        pipeline._backfill_result_evidence(
            result=result,
            intel_results=intel_results,
            fundamental_context=fundamental_context,
            stock_name="贵州茅台",
        )

        intel = result.dashboard["intelligence"]
        self.assertIn("【依据：", intel["latest_news"])
        self.assertIn("【依据：", intel["earnings_outlook"])
        self.assertIn("净利润同比增长约20%", intel["earnings_outlook"])
        self.assertIn("【依据：", intel["sentiment_summary"])
        self.assertIn("正向线索", intel["sentiment_summary"])
        self.assertIn("【依据：", result.analysis_summary)
        self.assertIn("【依据：", result.news_summary)
        self.assertIn("【依据：", result.market_sentiment)
        self.assertIn("搜索来源：", result.data_sources)
        self.assertIn("latest_news=Bocha", result.data_sources)
        self.assertIn("基本面来源：", result.data_sources)
        self.assertIn("引用条目：", result.data_sources)
        self.assertIn("【依据：", result.dashboard["core_conclusion"]["one_sentence"])

    def test_backfill_replaces_uncited_claims_with_missing_notes_when_no_evidence(self) -> None:
        """Unsupported qualitative claims should become explicit missing-data notes."""
        pipeline, AnalysisResult = self._load_pipeline_types()
        result = AnalysisResult(
            code="600519",
            name="贵州茅台",
            sentiment_score=55,
            trend_prediction="震荡",
            operation_advice="观望",
            analysis_summary="市场情绪中性",
            news_summary="近期无重大消息",
            market_sentiment="市场情绪中性",
            fundamental_analysis="基本面稳健",
            dashboard={
                "intelligence": {
                    "latest_news": "近期无重大消息",
                    "earnings_outlook": "业绩预期改善",
                    "sentiment_summary": "市场情绪中性",
                }
            },
            data_sources="技术面数据",
        )

        pipeline._backfill_result_evidence(
            result=result,
            intel_results={},
            fundamental_context=None,
            stock_name="贵州茅台",
        )

        intel = result.dashboard["intelligence"]
        self.assertTrue(intel["latest_news"].startswith("信息缺失："))
        self.assertTrue(intel["earnings_outlook"].startswith("信息缺失："))
        self.assertTrue(intel["sentiment_summary"].startswith("信息缺失："))
        self.assertTrue(result.news_summary.startswith("信息缺失："))
        self.assertTrue(result.market_sentiment.startswith("信息缺失："))
        self.assertTrue(result.fundamental_analysis.startswith("信息缺失："))
        self.assertTrue(result.analysis_summary.startswith("信息缺失："))
        self.assertTrue(result.data_sources.startswith("信息缺失："))


if __name__ == "__main__":
    unittest.main()
