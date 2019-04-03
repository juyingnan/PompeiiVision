import os
import shutil


def folder_match(src_root_path, dst_root_path, is_read_only=False):
    src_file_name_list = [file_name for file_name in os.listdir(src_root_path) if
                          os.path.isfile(os.path.join(src_root_path, file_name))]
    dst_file_name_list = [file_name for file_name in os.listdir(dst_root_path) if
                          os.path.isfile(os.path.join(dst_root_path, file_name))]
    suc_count = 0
    fail_count = 0
    print('matching files from %s to %s' % (src_root_path, dst_root_path))
    for dst_file in dst_file_name_list:
        if dst_file in src_file_name_list:
            suc_count += 1
            src_file_path = src_root_path + dst_file
            dst_file_path = dst_root_path + dst_file
            if not is_read_only:
                shutil.copy(src_file_path, dst_file_path)
        else:
            fail_count += 1
            print('Failed: %s' % (dst_root_path + dst_file))
    print('Total: %d; Success: %d; Failed: %d' % (len(dst_file_name_list), suc_count, fail_count))


for i in range(4):
    dst_root = r'D:\Projects\pompeii\20190319\svd_test_raw\%d/' % (i + 1)
    src_root = r'D:\Projects\pompeii\20190319\raw_all/'
    folder_match(src_root, dst_root, is_read_only=False)
