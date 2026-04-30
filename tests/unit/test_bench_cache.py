from benchmarks.cache import IngestionCache, dataset_hash, snapshot_path


def test_dataset_hash_stable(tmp_path):
    p = tmp_path / "oracle.json"
    p.write_text('[{"question_id": "q1"}]')
    assert dataset_hash(str(p)) == dataset_hash(str(p))


def test_dataset_hash_changes_with_content(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text('[{"q": 1}]')
    b.write_text('[{"q": 2}]')
    assert dataset_hash(str(a)) != dataset_hash(str(b))


def test_snapshot_path_includes_hash_and_mode(tmp_path):
    sp = snapshot_path(dataset_hash="abc123", extract_mode=True, cache_dir=tmp_path)
    assert "abc123" in str(sp)
    assert "extract" in str(sp)
    sp2 = snapshot_path(dataset_hash="abc123", extract_mode=False, cache_dir=tmp_path)
    assert "chunks" in str(sp2)
    assert sp != sp2


def test_ingestion_cache_round_trip(tmp_path):
    cache = IngestionCache(cache_dir=tmp_path)
    src = tmp_path / "src.db"
    src.write_bytes(b"fake sqlite bytes")
    assert not cache.has(dataset_hash="h1", extract_mode=False)
    cache.save(src_db_path=src, dataset_hash="h1", extract_mode=False)
    assert cache.has(dataset_hash="h1", extract_mode=False)
    out = tmp_path / "restored.db"
    cache.restore(out_path=out, dataset_hash="h1", extract_mode=False)
    assert out.read_bytes() == b"fake sqlite bytes"


def test_ingestion_cache_modes_dont_collide(tmp_path):
    cache = IngestionCache(cache_dir=tmp_path)
    src = tmp_path / "src.db"
    src.write_bytes(b"x")
    cache.save(src_db_path=src, dataset_hash="h", extract_mode=False)
    assert cache.has(dataset_hash="h", extract_mode=False)
    assert not cache.has(dataset_hash="h", extract_mode=True)
