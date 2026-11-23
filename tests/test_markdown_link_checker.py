from tools.docs.check_markdown_links import check_document, references, url_path


def test_url_parent_normalization_precedes_symlink_resolution(tmp_path):
    home = tmp_path / "home"
    actual = tmp_path / "data" / "campaign"
    (actual / "reports" / "final").mkdir(parents=True)
    (actual / "assessment").mkdir()
    (actual / "assessment" / "decision.json").write_text("{}")
    home.mkdir()
    alias = home / "review"
    alias.symlink_to(actual / "reports" / "final", target_is_directory=True)
    document = alias / "report.md"
    document.write_text("[Decision](../../assessment/decision.json)\n")
    assert (alias / "../../assessment/decision.json").exists()
    assert not url_path(document, "../../assessment/decision.json").exists()
    result = check_document(document, tmp_path)
    assert result["issues"][0]["reason"] == "URL-normalized destination is missing"


def test_portable_report_checks_images_reference_links_and_fragments(tmp_path):
    (tmp_path / "figure.png").write_bytes(b"example")
    document = tmp_path / "report.md"
    document.write_text("# Evidence\n\n![Figure][preview]\n\n[Jump](#evidence)\n\n"
                        "[preview]: figure.png\n\n```text\n[not a link](missing.md)\n```\n")
    assert not check_document(document, tmp_path, portable=True)["issues"]
    urls, anchors = references(document.read_text())
    assert urls == ["figure.png", "#evidence"] and anchors == {"evidence"}
    document.write_text("[Bad fragment](#absent)\n")
    assert check_document(document, tmp_path)["issues"][0]["reason"] == "Markdown heading fragment is missing"


def test_portable_report_rejects_machine_specific_asset_link(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "figure.png").write_bytes(b"example")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "assets").symlink_to(data, target_is_directory=True)
    document = repo / "report.md"
    document.write_text("![Figure](assets/figure.png)\n")
    assert not check_document(document, repo)["issues"]
    assert check_document(document, repo, portable=True)["issues"]


def test_portable_report_requires_relative_local_urls(tmp_path):
    target = tmp_path / "evidence.md"
    target.write_text("# Evidence\n")
    document = tmp_path / "report.md"
    for destination in [str(target), target.as_uri()]:
        document.write_text(f"[Evidence]({destination})\n")
        assert not check_document(document, tmp_path)["issues"]
        assert check_document(document, tmp_path, portable=True)["issues"][0]["reason"] == (
            "portable document uses an absolute local URL"
        )


def test_portable_document_must_be_physical_and_inside_root_even_without_links(tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("# Evidence\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    alias = repo / "INDEX.md"
    alias.symlink_to(outside)
    for document in [outside, alias]:
        assert not check_document(document, repo)["issues"]
        assert check_document(document, repo, portable=True)["issues"][0]["reason"] == (
            "portable Markdown document is outside the checkout or uses a symlink"
        )
