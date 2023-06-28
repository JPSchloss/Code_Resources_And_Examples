import os
import datetime
import pprint

def format_timestamp(timestamp):
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

def get_file_metadata(file_path):
    metadata = {}
    file_stats = os.stat(file_path)
    
    metadata['File Size in KB'] = file_stats.st_size / (1024)
    metadata['File Size in MB'] = file_stats.st_size / (1024 * 1024)
    metadata['File Format'] = os.path.splitext(file_path)[1]
    metadata['Last Access Time'] = format_timestamp(file_stats.st_atime)
    metadata['Last Metadata Modification'] = format_timestamp(file_stats.st_ctime)
    metadata['Last Content Modification'] = format_timestamp(file_stats.st_mtime)
    
    return metadata

file_path = 'path/to/your/file.csv'
file_metadata = get_file_metadata(file_path)
pprint.pprint(file_metadata)