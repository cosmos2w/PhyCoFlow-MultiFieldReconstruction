from tools.docs.check_workspace_layout import check_layout


def test_retired_review_alias_is_rejected_without_touching_its_target(tmp_path):
    for name in ("_Plans", "_Reports"):
        (tmp_path / "UpdatePlans" / name).mkdir(parents=True)
    report = tmp_path / "payload"
    report.mkdir()
    (report / "report.md").write_text("# Evidence\n")
    alias = tmp_path / "Future_Campaign_Final_Review"
    alias.symlink_to(report, target_is_directory=True)
    assert len(check_layout(tmp_path)) == 1
    assert (report / "report.md").read_text() == "# Evidence\n"
    alias.rename(tmp_path / "UpdatePlans/_Reports" / "_bk")
    # A current library cannot itself contain a symlink archive entrance.
    assert len(check_layout(tmp_path)) == 1


def test_current_library_keeps_archive_aliases_below_bk(tmp_path):
    for name in ("_Plans", "_Reports"):
        library = tmp_path / "UpdatePlans" / name
        (library / "_bk/Legacy_Aliases").mkdir(parents=True)
        (library / "README.md").write_text("# Library\n")
        (library / "Current_Campaign").mkdir()
        (library / "_bk/Legacy_Aliases/old").symlink_to(tmp_path)
    assert not check_layout(tmp_path)
    (tmp_path / "UpdatePlans/_Reports/old.md").write_text("# Old report\n")
    assert len(check_layout(tmp_path)) == 1
