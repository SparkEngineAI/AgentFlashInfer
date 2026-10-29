import os

import shutil
 
from agent_infer.unified_store.file_store import FileStoreClient
 
 
class TestFileStore:
    @classmethod
    def setup_method(self):
        self.workspace = os.path.dirname(__file__) + '/test_work_space'
        os.makedirs(self.workspace)
 
    @classmethod
    def teardown_method(self):
        shutil.rmtree(os.path.dirname(__file__) + '/test_work_space')
 
    def test_basic_func(self):
        store = FileStoreClient(self.workspace)
        content = 'you are a good man\n'
        session_id = 'session_id1'
 
        file_path = f'{session_id}/test.text'
        try:
            # create a file and write content
            store.create(file_path, content)
 
            store.write(file_path, "i am a good man\n")
            # content should be 'i am a good man\n'
            content = store.read(file_path)
            assert content == 'i am a good man\n'
 
            lines = ['you are a good man\n', "i am a good man\n", 'yes\n']
            store.write_lines(file_path, lines)
 
            # contents and lines should be equal
            contents = store.read_lines(file_path)
            assert contents == lines
 
            store.append(file_path, 'no\n')
 
            contents = store.read_lines(file_path)
            assert len(contents) == 4
 
            files = store.list_files(self.workspace, recursive=True)
            print(files)
 
            store.delete(file_path)
 
        except Exception as e:
            assert False