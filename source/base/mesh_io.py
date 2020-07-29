import numpy as np
import scipy.sparse as sparse
from source.base import utils
from source.base import file_utils


def read_off(file_path):
    """
    Read .off file.
    :param file_path:
    :return: vertices, faces, vertex_colors
    """

    def skip_empty_lines(file):
        cursor = file.tell()
        line = file.readline()
        while line:
            line = line.strip()
            if not line.startswith('#') and line != '':
                break

            cursor = file.tell()
            line = file.readline()

        file.seek(cursor)

    with open(file_path, 'r') as fp:
        header = fp.readline().strip()
        if header != 'OFF' and header != 'COFF':
            raise ValueError('Not a valid OFF header')

        has_color = header == 'COFF'

        skip_empty_lines(fp)
        
        # get number of elements
        n_vertices = 0
        n_faces = 0
        n_edges = 0
        has_sizes = False
        while not has_sizes:
            line = fp.readline()
            if not line:
                break
            else:
                line = line.strip()

            if not has_sizes:
                n_vertices, n_faces, n_edges = tuple([int(s) for s in line.strip().split(' ')])
                has_sizes = True

        if not has_sizes:
            raise ValueError('Element count not found')

        skip_empty_lines(fp)

        # get vertices and faces
        vertices = [[float(s.strip()) for s in fp.readline().strip().split(' ')
                     if s.strip() != ''] for i_vert in range(n_vertices)]
        faces = [[int(s.strip()) for s in fp.readline().strip().split(' ')
                  if s.strip() != ''][1:] for i_face in range(n_faces)]

        vertices = np.asarray(vertices).astype(np.float32)
        faces = np.asarray(faces)

        if has_color:
            vertex_colors = vertices[:, [3, 4, 5]]
            vertices = vertices[:, [0, 1, 2]]
        else:
            vertex_colors = None

    return vertices, faces, vertex_colors


def write_off(file_path: str, vertices: np.ndarray, faces: np.ndarray,
              colors_vertex: np.ndarray = np.array([]), colors_face: np.ndarray = np.array([])) -> None:
    """
    Write mesh as .off file.
    :param file_path:
    :param vertices:
    :param faces:
    :param colors_vertex:
    :param colors_face:
    :return:
    """
    
    if len(vertices) == 0:
        return

    file_utils.make_dir_for_file(file_path)
    
    with open(file_path, 'w') as fp:  
        
        # write header
        if colors_face.size == 0 and colors_vertex.size == 0:
            fp.write("OFF\n")
        else:
            fp.write("COFF\n")
        
        # convert 2d vertices to 3d
        if len(vertices[0]) == 2:
            vertices_2p5d = np.zeros((len(vertices), 3))
            vertices_2p5d[:, :2] = vertices
            vertices_2p5d[:, 2] = 0.0
            vertices = vertices_2p5d

        # write number of elements
        fp.write(str(len(vertices)) + " " + str(len(faces)) + " 0\n")  # 0 for optional #edges
        
        # write vertices
        if colors_vertex.size == 0:
            for v in vertices:
                fp.write(str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + "\n")
        else:
            color_channels = colors_vertex.shape[1]
            for vi, v in enumerate(vertices):
                line_vertex = str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + " "
                # line_vertex = line_vertex + str(color_channels) + " "
                for c in range(color_channels):
                    line_vertex += str(colors_vertex[vi][c]) + " "
                fp.write(line_vertex + "\n")

        # write faces
        if colors_face.size == 0:
            for f in faces:
                fp.write(str(len(f)) + " " + str(f[0]) + " " + str(f[1]) + " " + str(f[2]) + "\n")
        else:
            color_channels = colors_face.shape[1]
            for fi, f in enumerate(faces):
                line_face = str(len(f)) + " " + str(f[0]) + " " + str(f[1]) + " " + str(f[2]) + " "
                # line_face = line_face + str(color_channels) + " " # not for meshlab
                for c in range(color_channels):
                    line_face += str(colors_face[fi][c]) + " "
                fp.write(line_face + "\n")


def degenerated_to_slim_faces(faces, vertices):
    eps = 0.0001
    
    def update_face(f, vertices):
        if f[0] != f[1] and f[0] != f[2] and f[1] != f[2]:  # all vertices are different
            return f, vertices
        else:  # some vertices are the same
            v_new = np.expand_dims(vertices[f[0]], axis=0) + eps
            vertices = np.concatenate([vertices, v_new], axis=0)
            if f[0] != f[1]:  # v2 is a duplicate
                return [f[0], f[1], vertices.shape[0]-1], vertices
            elif f[0] != f[2]:  # v1 is a duplicate
                return [f[0], vertices.shape[0]-1, f[2]], vertices
            elif f[1] != f[2]:  # v0 is a duplicate
                return [vertices.shape[0]-1, f[1], f[2]], vertices
            else:  # all vertices are the same
                v_new_2 = np.expand_dims(vertices[f[0]], axis=0) - eps
                vertices = np.concatenate([vertices, v_new_2], axis=0)
                return [f[0], vertices.shape[0]-2, vertices.shape[0]-1], vertices
    
    for fi, f in enumerate(faces):
        faces[fi], vertices = update_face(f, vertices)
    
    return faces, vertices


def clean_mesh(faces, vertices):
    # remove all unnecessary vertices and update faces
    faces_flattened = [item for sublist in faces for item in sublist]
    vertices = [vertices[vi] for vi in faces_flattened]
    faces = [[fi*3, fi*3+1, fi*3+2] for fi, vi in enumerate(faces)]
        
    return faces, vertices                        


def load_mesh(file):
    vertices, faces, v_col = read_off(file)

    # https://stackoverflow.com/questions/537086/reserve-memory-for-list-in-python
    # pre-allocated list
    num_entries = len(faces) * 2 * 3  # faces * circle_directions * triangle_vertices
    v_a = [None]*num_entries
    v_b = [None]*num_entries
    val = [None]*num_entries
    for face_id, face in enumerate(faces):
        face_circular = face.copy()
        face_circular.append(face[0])
        
        for i in range(len(face)):
            v_a[face_id * 3 * 2 + i] = face_circular[i]
            v_b[face_id * 3 * 2 + i] = face_circular[i+1]
            val[face_id * 3 * 2 + i] = np.float32(1)  # float32 because torch doesn't support bool tensors
            v_a[face_id * 3 * 2 + 3 + i] = face_circular[i+1]
            v_b[face_id * 3 * 2 + 3 + i] = face_circular[i]
            val[face_id * 3 * 2 + 3 + i] = np.float32(1)

    adj_mat = sparse.csr_matrix((val, (v_a, v_b)), [vertices.shape[0], vertices.shape[0]])

    adj_mat[adj_mat > np.float32(1)] = np.float32(1)  # csr sums the values -> can lead to something else than 0 and 1

    if not utils.is_matrix_symmetric(adj_mat):
        print('WARING: loaded adjacency matrix of ' + file + ' is NOT symmetric')

    return vertices, adj_mat

