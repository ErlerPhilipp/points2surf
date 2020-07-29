import numpy as np
import os
import scipy.sparse as sparse


def filename_to_hash(file_path):
    import hashlib
    if not os.path.isfile(file_path):
        raise ValueError('Path does not point to a file: {}'.format(file_path))
    hash_input = os.path.basename(file_path).split('.')[0]
    hash = int(hashlib.md5(hash_input.encode()).hexdigest(), 16) % (2**32 - 1)
    return hash


def load_npy_if_valid(filename, data_type, mmap_mode=None):
    if not os.path.isfile(filename) or (os.path.isfile(filename + '.npy') and
                                        (os.path.getmtime(filename + '.npy') > os.path.getmtime(filename))):
        data = np.load(filename + '.npy', mmap_mode).astype(data_type)
    else:
        data = np.loadtxt(filename).astype(data_type)
        np.save(filename + '.npy', data)
        if os.path.isfile(filename + '.npy') and (os.path.getmtime(filename + '.npy') < os.path.getmtime(filename)):
            print('Warning: \"' + filename + '\" is newer than \"' + filename + '.npy\". Loading \"' + filename + '\"')

    return data


def npz_to_txt(path_in, path_out, num_files=None):
    
    files = [f for f in os.listdir(path_in) if os.path.isfile(os.path.join(path_in, f)) and f[-4:] == '.npz']
    
    for fi, f in enumerate(files):
        print('Converting npz to txt: ' + f)
        npz_to_txt_file(file_npz_in=os.path.join(path_in, f), file_txt_out=os.path.join(path_out, f[:-4]))
        if not num_files is None and fi >= num_files - 1:
            break


def npz_to_txt_file(file_npz_in, file_txt_out):
    
    sparse_mat = sparse.load_npz(file_npz_in)
    
    coo = sparse_mat.nonzero()
    coo_x = coo[0]
    coo_y = coo[1]
    
    make_dir_for_file(file_txt_out)
            
    with open(file_txt_out, 'w') as the_file:
        for i in range(coo_x.shape[0]):
            the_file.write(str(coo_x[i]) + ' ' + str(coo_y[i]) + ' ' + str(sparse_mat[coo_x[i], coo_y[i]]) + '\n')


def txt_to_npz_file(file_txt_in, file_npz_out, dtype=None, size=None):
        if dtype is None:
            dtype={'names': ('i', 'j', 'val'),
                    'formats': (np.uint32, np.uint32, np.float32)}
        v_from, v_to, val = np.loadtxt(file_txt_in, unpack=True, dtype=dtype)
        if size is None:
            size = max(v_from.max(), v_to.max())
        sparse_mat = sparse.coo_matrix((val, (v_from, v_to)), (size+1, size+1)).tocsr()
        sparse.save_npz(file_npz_out, sparse_mat)


def txt_to_npz(path, ending='.txt', dtype=None, size=None):

    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f[-len(ending):] == ending]
    
    for f in files:
        file = os.path.join(path, f)
        file_npz = file+'.npz'
        print(file + ' to ' + file_npz)
        txt_to_npz_file(file_txt_in=file, file_npz_out=file_npz, dtype=dtype, size=size)


def txt_to_npy_file(file_txt_in, file_npy_out):
    arr = np.loadtxt(file_txt_in, unpack=True)
    arr = arr.transpose()[:, :3].astype(np.float32)
    np.save(file_npy_out, arr)


def txt_to_npy(path, ending='.txt'):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f[-len(ending):] == ending]

    for f in files:
        file = os.path.join(path, f)
        file_npy = file + '.npy'
        print(file + ' to ' + file_npy)
        txt_to_npy_file(file_txt_in=file, file_npy_out=file_npy)


def concat_txt_files(files_in, file_out):
    lines_per_file = []
    for fi, f in enumerate(files_in):
        with open(f) as file:
            new_lines = file.readlines()
            new_lines = [l.replace(' \n', '') for l in new_lines]
            lines_per_file.append(new_lines)

    # assume same number of lines in all files
    lines_output = []
    for li in range(len(lines_per_file[0])):
        lines = [f[li] for f in lines_per_file]
        lines_output.append(' '.join(lines))

    with open(file_out, "w+") as file:
        file.writelines(lines_output)


def concat_txt_dirs(ref_dir, ref_ending, dirs, endings_per_dir=('.txt',), out_dir='../concat/', out_ending='.txt'):

    file_stems = [os.path.splitext(f)[0] for f in os.listdir(ref_dir)
                  if os.path.isfile(os.path.join(ref_dir, f)) and f[-len(ref_ending):] == ref_ending]
    files = []
    for fi, file_stem in enumerate(file_stems):
        files.append([os.path.join(dir, file_stem + endings_per_dir[di]) for di, dir in enumerate(dirs)])

    os.makedirs(out_dir, exist_ok=True)

    for fi, f in enumerate(files):
        file_out = os.path.join(out_dir, file_stems[fi] + out_ending)
        if call_necessary(f, file_out):
            print('concat {} to {}'.format(f, file_out))
            concat_txt_files(files_in=f, file_out=file_out)


def make_dir_for_file(file):
    file_dir = os.path.dirname(file)
    if file_dir != '':
        if not os.path.exists(file_dir):
            try:
                os.makedirs(os.path.dirname(file))
            except OSError as exc: # Guard against race condition
                raise


def load_npz(npz_file, mmap_mode=None):
    try:
        return sparse.load_npz(npz_file)
    except:
        # npz does not contain a sparse matrix but the data to construct one
        geodesic_file = np.load(npz_file, mmap_mode)
        data = geodesic_file['data']
        col_ind = geodesic_file['col_ind']
        row_ind = geodesic_file['row_ind']
        shape = tuple(geodesic_file['shape'])
        return sparse.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def path_leaf(path):
    import ntpath
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def touch_files_in_dir(dir, extension=None):
    import os
    from pathlib import Path

    files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    if extension is not None:
        files = [f for f in files if f[-len(extension):] == extension]
    for fi, f in enumerate(files):
        file_in_abs = os.path.join(dir, f)
        Path(file_in_abs).touch()


def copy_list_of_files_in_dir(dir_in, dir_out, file_list):
    import os
    import shutil

    files = [f for f in os.listdir(dir_in) if os.path.isfile(os.path.join(dir_in, f))]
    file_stems = [os.path.basename(f) for f in files]
    file_stems = [f.split('.')[0] for f in file_stems]

    if file_list is None:
        files_to_copy_set = set(file_stems)  # set for efficient search
    else:
        with open(file_list) as f:
            files_to_copy_set = f.readlines()
            files_to_copy_set = [f.replace('\n', '') for f in files_to_copy_set]
            files_to_copy_set = [f.split('.')[0] for f in files_to_copy_set]
            files_to_copy_set = set(files_to_copy_set)

    os.makedirs(dir_out, exist_ok=True)

    for fi, f in enumerate(files):
        if file_stems[fi] in files_to_copy_set:
            file_in_abs = os.path.join(dir_in, f)
            file_out_abs = os.path.join(dir_out, f)
            shutil.copyfile(src=file_in_abs, dst=file_out_abs)


def call_necessary(file_in, file_out, min_file_size=0):
    """
    Check if all input files exist and at least one output file does not exist or is invalid.
    :param file_in: list of str or str
    :param file_out: list of str or str
    :param min_file_size: int
    :return:
    """

    if isinstance(file_in, str):
        file_in = [file_in]
    elif isinstance(file_in, list):
        pass
    else:
        raise ValueError('Wrong input type')

    if isinstance(file_out, str):
        file_out = [file_out]
    elif isinstance(file_out, list):
        pass
    else:
        raise ValueError('Wrong output type')

    inputs_missing = [f for f in file_in if not os.path.isfile(f)]
    if len(inputs_missing) > 0:
        print('WARNING: Input file are missing: {}'.format(inputs_missing))
        return False

    outputs_missing = [f for f in file_out if not os.path.isfile(f)]
    if len(outputs_missing) > 0:
        if len(outputs_missing) < len(file_out):
            print("WARNING: Only some output files are missing: {}".format(outputs_missing))
        return True

    min_output_file_size = min([os.path.getsize(f) for f in file_out])
    if min_output_file_size < min_file_size:
        return True

    oldest_input_file_mtime = max([os.path.getmtime(f) for f in file_in])
    youngest_output_file_mtime = min([os.path.getmtime(f) for f in file_out])

    if oldest_input_file_mtime >= youngest_output_file_mtime:
        # debug
        import time
        input_file_mtime_arg_max = np.argmax(np.array([os.path.getmtime(f) for f in file_in]))
        output_file_mtime_arg_min = np.argmin(np.array([os.path.getmtime(f) for f in file_out]))
        input_file_mtime_max = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(oldest_input_file_mtime))
        output_file_mtime_min = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(youngest_output_file_mtime))
        print('Input file {} \nis newer than output file {}: \n{} >= {}'.format(
            file_in[input_file_mtime_arg_max], file_out[output_file_mtime_arg_min],
            input_file_mtime_max, output_file_mtime_min))
        return True

    return False


def xyz_to_npy(file):
    from source.base import point_cloud
    p = point_cloud.load_xyz(file)
    np.save(file + '.npy', p)

