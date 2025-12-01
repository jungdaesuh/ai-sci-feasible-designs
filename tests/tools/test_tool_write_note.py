"""Test the write_note tool."""

from ai_scientist import memory
from ai_scientist.tools.integration import write_note


def test_write_note_creates_file_and_db_entry(tmp_path):
    """Verify that write_note creates a file and logs to the world model."""

    notes_dir = tmp_path / "notes"
    db_path = tmp_path / "wm.sqlite"
    content = "# Analysis\nThis is a test note."

    msg = write_note(
        content,
        filename="test_note.md",
        out_dir=notes_dir,
        memory_db=db_path,
        experiment_id=1,
        cycle=1,
    )

    expected_path = notes_dir / "test_note.md"
    assert expected_path.exists()
    assert expected_path.read_text(encoding="utf-8") == content
    assert "saved to" in msg

    with memory.WorldModel(db_path) as world_model:
        notes = world_model.notes(1, 1)
        assert len(notes) == 1
        assert notes[0]["content"] == content


def test_write_note_auto_filename_and_db(tmp_path):
    """Verify auto-generated filename and DB persistence."""

    notes_dir = tmp_path / "notes"
    db_path = tmp_path / "wm.sqlite"
    content = "Unique content"

    write_note(
        content,
        out_dir=notes_dir,
        memory_db=db_path,
        experiment_id=2,
        cycle=1,
    )

    files = list(notes_dir.glob("note_*.md"))
    assert len(files) == 1
    assert files[0].read_text(encoding="utf-8") == content

    with memory.WorldModel(db_path) as world_model:
        notes = world_model.notes(2, 1)
        assert len(notes) == 1
        assert notes[0]["content"] == content
