"""
Performance data: the only part of this app that measures rather than predicts.

Everything here runs against recorded API payloads. There is no channel
connected to this machine, so the live request path is unproven — the same
position the YouTube Data API was in before a key arrived. What *is* tested is
everything that happens to a response once it exists: the column-header
unpacking, the metric-refusal fallback, the join back to the ledger, where
three seconds falls on a retention curve, and the refusal to report a
correlation from too few videos.

The shape of the recorded payloads is the contract. If Google changes it these
tests keep passing and the live call starts failing, which is worth stating
plainly rather than pretending otherwise.
"""
from __future__ import annotations

import pytest

import analytics_engine as ae


# ---------------------------------------------------------------------------
# Recorded payloads
# ---------------------------------------------------------------------------

VIDEO_REPORT = {
    "kind": "youtubeAnalytics#resultTable",
    "columnHeaders": [
        {"name": "video", "dataType": "STRING"},
        {"name": "views", "dataType": "INTEGER"},
        {"name": "estimatedMinutesWatched", "dataType": "INTEGER"},
        {"name": "averageViewDuration", "dataType": "INTEGER"},
        {"name": "averageViewPercentage", "dataType": "FLOAT"},
        {"name": "subscribersGained", "dataType": "INTEGER"},
        {"name": "likes", "dataType": "INTEGER"},
        {"name": "comments", "dataType": "INTEGER"},
        {"name": "shares", "dataType": "INTEGER"},
    ],
    "rows": [
        ["vidAAAAAAAA", 41203, 3891, 34, 48.6, 312, 2104, 88, 140],
        ["vidBBBBBBBB", 8800, 690, 28, 40.0, 41, 302, 12, 19],
        ["vidCCCCCCCC", 1502, 97, 21, 30.0, 6, 44, 3, 2],
    ],
}

RETENTION = {
    "columnHeaders": [
        {"name": "elapsedVideoTimeRatio", "dataType": "FLOAT"},
        {"name": "audienceWatchRatio", "dataType": "FLOAT"},
    ],
    # Deliberately out of order: the API does not promise a sort.
    "rows": [[0.10, 0.71], [0.00, 1.00], [0.50, 0.44], [0.04, 0.83], [1.00, 0.21]],
}

LEDGER = [
    {"video_name": "minimalist_1.mp4", "duration": 70.0, "script": "x " * 40,
     "youtube_video_id": "vidAAAAAAAA", "youtube_title": "Why consistency wins"},
    {"video_name": "minimalist_2.mp4", "duration": 68.0, "script": "y " * 40,
     "youtube_video_id": "vidBBBBBBBB", "youtube_title": "The cost of waiting"},
    # Rendered but never uploaded: no id, so it cannot be measured.
    {"video_name": "minimalist_3.mp4", "duration": 66.0, "script": "z " * 40},
]


class FakeReports:
    def __init__(self, payload, fail_on=()):
        self.payload = payload
        self.fail_on = set(fail_on)
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        metrics = set(str(kwargs.get("metrics") or "").split(","))
        if metrics & self.fail_on:
            raise RuntimeError(
                "<HttpError 400 ... 'Unknown metric impressions'>")
        outer = self

        class _Exec:
            def execute(self):
                return outer.payload

        return _Exec()


class FakeService:
    def __init__(self, payload, fail_on=()):
        self._reports = FakeReports(payload, fail_on)

    def reports(self):
        return self._reports


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

class TestParsing:

    def test_headers_and_rows_become_dicts(self):
        rows = ae.parse_report(VIDEO_REPORT)
        assert len(rows) == 3
        assert rows[0]["video"] == "vidAAAAAAAA"
        assert rows[0]["views"] == 41203
        assert rows[0]["averageViewPercentage"] == 48.6

    @pytest.mark.parametrize("payload", [
        {}, None, "nonsense", {"rows": [[1, 2]]}, {"columnHeaders": []},
    ])
    def test_a_malformed_response_is_empty_not_an_exception(self, payload):
        assert ae.parse_report(payload) == []

    def test_a_row_that_is_not_a_row_is_skipped(self):
        payload = dict(VIDEO_REPORT, rows=[["vidAAAAAAAA", 1, 2, 3, 4, 5, 6, 7, 8],
                                           "not a row", None])
        assert len(ae.parse_report(payload)) == 1


# ---------------------------------------------------------------------------
# The metric-refusal fallback
# ---------------------------------------------------------------------------

class TestOptionalMetrics:

    def test_it_asks_for_impressions_and_ctr_first(self):
        service = FakeService(VIDEO_REPORT)
        ae.video_report(["vidAAAAAAAA"], client=service)
        asked = service.reports().calls[0]["metrics"]
        assert "impressions" in asked
        assert "impressionClickThroughRate" in asked

    def test_a_refused_metric_retries_with_the_core_set(self):
        """Impressions and CTR are Studio-only on this API. Asking for them
        unconditionally takes the whole report down with a 400."""
        service = FakeService(VIDEO_REPORT, fail_on={"impressions"})
        report = ae.video_report(["vidAAAAAAAA"], client=service)

        assert len(service.reports().calls) == 2
        assert report["rows"], "the retry returned nothing"
        assert set(report["unavailable"]) == set(ae.OPTIONAL_METRICS)
        assert "impressions" not in service.reports().calls[1]["metrics"]

    def test_the_result_says_which_numbers_were_not_served(self):
        """A silent zero would read as 'nobody saw it' rather than 'YouTube
        does not tell us'."""
        service = FakeService(VIDEO_REPORT, fail_on={"impressions"})
        report = ae.video_report([], client=service)
        assert report["unavailable"] == list(ae.OPTIONAL_METRICS)

    def test_a_real_failure_is_not_swallowed_as_a_metric_problem(self):
        class Broken:
            def reports(self):
                class R:
                    def query(self, **_kwargs):
                        raise RuntimeError("<HttpError 403 forbidden>")
                return R()

        with pytest.raises(ae.AnalyticsError):
            ae.video_report([], client=Broken())

    def test_the_video_filter_is_capped(self):
        """The filter parameter caps at 500 characters; 40 ids is inside it."""
        service = FakeService(VIDEO_REPORT)
        ae.video_report([f"vid{i:08d}" for i in range(200)], client=service)
        sent = service.reports().calls[0]["filters"]
        assert len(sent) < 500
        assert sent.count(",") <= 39


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------

class TestRetention:

    def test_the_curve_comes_back_sorted(self):
        curve = ae.retention_curve("vidAAAAAAAA", client=FakeService(RETENTION))
        ratios = [point["ratio"] for point in curve]
        assert ratios == sorted(ratios)
        assert curve[0]["watching"] == 1.0

    def test_three_seconds_lands_where_the_runtime_puts_it(self):
        """On a 70-second short the hook is 4.3% in, not 3% and not 30%."""
        curve = ae.retention_curve("v", client=FakeService(RETENTION))
        held = ae.hook_retention(curve, duration=70.0)
        # 3/70 = 0.043, so the first sample at or past it is the 0.10 point.
        assert held == pytest.approx(0.71)

    def test_a_shorter_video_samples_earlier_in_the_curve(self):
        curve = ae.retention_curve("v", client=FakeService(RETENTION))
        # On a 75s video 3s is 4%, which lands on the 0.04 sample.
        assert ae.hook_retention(curve, duration=75.0) == pytest.approx(0.83)

    @pytest.mark.parametrize("curve,duration", [
        ([], 70.0), ([{"ratio": 0.0, "watching": 1.0}], 0.0), ([], 0.0),
    ])
    def test_unknown_reads_as_zero_for_the_caller_to_interpret(self, curve, duration):
        assert ae.hook_retention(curve, duration) == 0.0

    def test_a_curve_that_ends_early_uses_its_last_point(self):
        curve = [{"ratio": 0.0, "watching": 1.0}, {"ratio": 0.01, "watching": 0.9}]
        assert ae.hook_retention(curve, duration=70.0) == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Joining performance to the render that produced it
# ---------------------------------------------------------------------------

class TestJoin:

    def test_only_uploaded_renders_are_measurable(self):
        assert len(ae.published_entries(LEDGER)) == 2

    def test_it_joins_on_the_video_id(self):
        report = {"rows": ae.parse_report(VIDEO_REPORT)}
        joined = ae.join_performance(LEDGER, report)
        assert [item["video_id"] for item in joined] == ["vidAAAAAAAA", "vidBBBBBBBB"]
        assert joined[0]["title"] == "Why consistency wins"
        assert joined[0]["views"] == 41203

    def test_uploads_this_tool_did_not_make_are_left_out(self):
        """vidCCCCCCCC is on the channel but not in the ledger. The point of
        the join is grading what this tool produced."""
        report = {"rows": ae.parse_report(VIDEO_REPORT)}
        joined = ae.join_performance(LEDGER, report)
        assert "vidCCCCCCCC" not in [item["video_id"] for item in joined]

    def test_it_sorts_by_views(self):
        report = {"rows": ae.parse_report(VIDEO_REPORT)}
        views = [item["views"] for item in ae.join_performance(LEDGER, report)]
        assert views == sorted(views, reverse=True)

    def test_a_render_with_no_reported_row_is_dropped(self):
        joined = ae.join_performance(LEDGER, {"rows": []})
        assert joined == []

    def test_the_summary_totals_what_was_joined(self):
        report = {"rows": ae.parse_report(VIDEO_REPORT)}
        summary = ae.summarise(ae.join_performance(LEDGER, report))
        assert summary["videos"] == 2
        assert summary["views"] == 41203 + 8800
        assert summary["subscribers"] == 312 + 41
        assert summary["best"]["video_id"] == "vidAAAAAAAA"

    def test_an_empty_summary_does_not_divide_by_zero(self):
        assert ae.summarise([])["videos"] == 0


# ---------------------------------------------------------------------------
# Grading the scorecard
# ---------------------------------------------------------------------------

def _joined(pairs):
    """Synthetic joined rows: (video_id, average_view_percent)."""
    return [{"video_id": vid, "average_view_percent": pct, "views": 100,
             "script": "", "duration": 70.0} for vid, pct in pairs]


class TestScoreVsRetention:

    def test_too_few_videos_says_so_rather_than_guessing(self):
        """A correlation over three uploads is noise with a decimal point."""
        joined = _joined([("a", 50.0), ("b", 40.0), ("c", 30.0)])
        verdict = ae.score_vs_retention(joined, {"a": 8.0, "b": 6.0, "c": 4.0})
        assert verdict["measured"] is False
        assert "3 of 5" in verdict["verdict"]

    def test_it_reports_agreement_when_the_score_tracks_retention(self):
        joined = _joined([("a", 60.0), ("b", 52.0), ("c", 44.0),
                          ("d", 36.0), ("e", 28.0)])
        scores = {"a": 9.0, "b": 8.0, "c": 7.0, "d": 6.0, "e": 5.0}
        verdict = ae.score_vs_retention(joined, scores)
        assert verdict["measured"] is True
        assert verdict["r"] > 0.9
        assert "predicting retention" in verdict["verdict"]

    def test_it_reports_an_inversion_plainly(self):
        """If the scorecard is backwards, saying so is the whole value."""
        joined = _joined([("a", 20.0), ("b", 30.0), ("c", 40.0),
                          ("d", 50.0), ("e", 60.0)])
        scores = {"a": 9.0, "b": 8.0, "c": 7.0, "d": 6.0, "e": 5.0}
        verdict = ae.score_vs_retention(joined, scores)
        assert verdict["r"] < -0.5
        assert "Inverted" in verdict["verdict"]
        assert "Trust the measurement" in verdict["verdict"]

    def test_identical_scores_are_not_a_correlation(self):
        joined = _joined([("a", 20.0), ("b", 30.0), ("c", 40.0),
                          ("d", 50.0), ("e", 60.0)])
        scores = {k: 7.0 for k in "abcde"}
        verdict = ae.score_vs_retention(joined, scores)
        assert verdict["measured"] is False
        assert "scored the same" in verdict["verdict"]

    def test_videos_with_no_measurement_are_not_counted(self):
        joined = _joined([("a", 0.0), ("b", 0.0), ("c", 40.0),
                          ("d", 50.0), ("e", 60.0)])
        scores = {k: float(i) for i, k in enumerate("abcde")}
        assert ae.score_vs_retention(joined, scores)["measured"] is False

    def test_the_wording_never_states_a_fact_about_the_world(self):
        """Five videos is a weak instrument. The verdict has to sound like it."""
        joined = _joined([(k, 30.0 + i * 8) for i, k in enumerate("abcde")])
        scores = {k: 5.0 + i for i, k in enumerate("abcde")}
        assert "so far" in ae.score_vs_retention(joined, scores)["verdict"]


# ---------------------------------------------------------------------------
# Scopes
# ---------------------------------------------------------------------------

class TestScopes:

    def test_the_analytics_scope_is_requested(self):
        import publisher

        assert ae.ANALYTICS_SCOPE in publisher.SCOPES

    def test_upload_still_works_the_same(self):
        """Adding a scope must not disturb publishing."""
        import publisher

        assert "https://www.googleapis.com/auth/youtube.upload" in publisher.SCOPES

    def test_a_token_without_the_new_scope_is_detected(self, tmp_path, monkeypatch):
        """A channel authorized before this existed uploads fine and cannot
        read reports. Without this check that surfaces as a 403 whose message
        never mentions the word 'scope'."""
        import json

        import publisher

        token = tmp_path / "youtube_token.json"
        token.write_text(json.dumps({
            "scopes": ["https://www.googleapis.com/auth/youtube.upload"]}),
            encoding="utf-8")
        monkeypatch.setattr(publisher, "TOKEN_PATH", str(token))

        assert ae.missing_scopes() == [ae.ANALYTICS_SCOPE]

    def test_a_full_token_needs_nothing(self, tmp_path, monkeypatch):
        import json

        import publisher

        token = tmp_path / "youtube_token.json"
        token.write_text(json.dumps({"scopes": list(publisher.SCOPES)}),
                         encoding="utf-8")
        monkeypatch.setattr(publisher, "TOKEN_PATH", str(token))

        assert ae.missing_scopes() == []

    def test_no_token_needs_everything(self, tmp_path, monkeypatch):
        import publisher

        monkeypatch.setattr(publisher, "TOKEN_PATH", str(tmp_path / "nope.json"))
        assert ae.missing_scopes() == list(publisher.SCOPES)

    def test_an_unreadable_token_is_treated_as_unauthorized(self, tmp_path, monkeypatch):
        import publisher

        token = tmp_path / "youtube_token.json"
        token.write_text("{ not json", encoding="utf-8")
        monkeypatch.setattr(publisher, "TOKEN_PATH", str(token))
        assert ae.missing_scopes() == list(publisher.SCOPES)


# ---------------------------------------------------------------------------
# The ledger link
# ---------------------------------------------------------------------------

class TestLedgerLink:

    def test_a_publish_can_be_recorded_against_its_render(self, tmp_path):
        import compliance

        compliance.append_ledger(str(tmp_path), {
            "video_name": "minimalist_9.mp4", "duration": 70.0, "script": "x"})
        updated = compliance.update_entry(str(tmp_path), "minimalist_9.mp4", {
            "youtube_video_id": "abc12345678", "youtube_title": "A title"})

        assert updated is not None
        assert updated["youtube_video_id"] == "abc12345678"
        # And it survives a reload, which is the point of a ledger.
        reloaded = compliance.load_ledger(str(tmp_path))
        assert reloaded[0]["youtube_video_id"] == "abc12345678"

    def test_updating_a_file_the_ledger_never_saw_returns_nothing(self, tmp_path):
        import compliance

        assert compliance.update_entry(str(tmp_path), "ghost.mp4", {"a": 1}) is None

    def test_the_update_does_not_disturb_other_entries(self, tmp_path):
        import compliance

        for index in range(3):
            compliance.append_ledger(str(tmp_path), {
                "video_name": f"reel_{index}.mp4", "duration": 30.0})
        compliance.update_entry(str(tmp_path), "reel_1.mp4", {"youtube_video_id": "z"})

        entries = compliance.load_ledger(str(tmp_path))
        assert len(entries) == 3
        assert "youtube_video_id" not in entries[0]
        assert entries[1]["youtube_video_id"] == "z"
        assert "youtube_video_id" not in entries[2]

    def test_the_upload_path_writes_the_link(self):
        """Without this line the ledger and the channel never join up."""
        import inspect

        import app

        source = inspect.getsource(app.render_atmosphere_publisher)
        assert "update_entry(" in source
        assert "youtube_video_id" in source
