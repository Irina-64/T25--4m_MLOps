def test_dag_file_contains_task_ids():
    path = 'dags/flight_pipeline.py'
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    for tid in ['download_data', 'preprocess', 'train', 'evaluate', 'register']:
        assert tid in content
