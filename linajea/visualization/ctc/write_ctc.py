import logging
import os

import numpy as np
import raster_geometry as rg
import tifffile

logger = logging.getLogger(__name__)

logging.basicConfig(level=20)

def write_ctc(graph, start_frame, end_frame, shape,
              out_dir, txt_fn, tif_fn, paint_sphere=False,
              voxel_size=None, gt=False):
    os.makedirs(out_dir, exist_ok=True)

    logger.info("writing frames %s,%s (graph %s)",
                start_frame, end_frame, graph.get_frames())
    # assert start_frame >= graph.get_frames()[0]
    if end_frame is None:
        end_frame = graph.get_frames()[1]
    # else:
    #     assert end_frame <= graph.get_frames()[1]
    track_cntr = 1

    node_to_track = {}
    curr_cells = set(graph.cells_by_frame(start_frame))
    for c in curr_cells:
        node_to_track[c] = (track_cntr, 0, start_frame)
        track_cntr += 1
    prev_cells = curr_cells

    cells_by_t_data = {}
    for f in range(start_frame+1, end_frame+1):
        cells_by_t_data[f] = []
        curr_cells = set(graph.cells_by_frame(f))
        for c in prev_cells:
            edges = list(graph.next_edges(c))
            assert len(edges) <= 2, "more than two children"
            if len(edges) == 1:
                e = edges[0]
                assert e[0] in curr_cells, "cell missing"
                assert e[1] in prev_cells, "cell missing"
                node_to_track[e[0]] = (
                    node_to_track[e[1]][0],
                    node_to_track[e[1]][1],
                    f)
                curr_cells.remove(e[0])
            elif len(edges) == 2:
                for e in edges:
                    assert e[0] in curr_cells, "cell missing"
                    assert e[1] in prev_cells, "cell missing"
                    node_to_track[e[0]] = (
                        track_cntr,
                        node_to_track[e[1]][0],
                        f)
                    track_cntr += 1
                    curr_cells.remove(e[0])

            if gt:
                for e in edges:
                    st = e[1]
                    nd = e[0]
                    dataSt = graph.nodes(data=True)[st]
                    dataNd = graph.nodes(data=True)[nd]
                    cells_by_t_data[f].append((
                        nd,
                        np.array([dataNd[d]
                                  for d in ['z', 'y', 'x']]),
                        np.array([dataSt[d] - dataNd[d]
                                  for d in ['z', 'y', 'x']])))
        for c in curr_cells:
            node_to_track[c] = (track_cntr, 0, f)
            track_cntr += 1
        prev_cells = set(graph.cells_by_frame(f))

    tracks = {}
    for c, v in node_to_track.items():
        if v[0] in tracks:
            tracks[v[0]][1].append(v[2])
        else:
            tracks[v[0]] = (v[1], [v[2]])

    if not gt:
        cells_by_t_data = {
            t: [
                (
                    cell,
                    np.array([data[d] for d in ['z', 'y', 'x']]),
                    np.array(data['parent_vector'])
                )
                for cell, data in graph.nodes(data=True)
                if 't' in data and data['t'] == t
            ]
            for t in range(start_frame, end_frame+1)
        }
    with open(os.path.join(out_dir, "parent_vectors.txt"), 'w') as of:
        for t, cs in cells_by_t_data.items():
            for c in cs:
                of.write("{} {} {} {} {} {} {}\n".format(
                    t, c[1][0], c[1][1], c[1][2],
                    c[2][0], c[2][1], c[2][2]))

    with open(os.path.join(out_dir, txt_fn), 'w') as of:
        for t, v in tracks.items():
            logger.debug("{} {} {} {}".format(
                t, min(v[1]), max(v[1]), v[0]))
            of.write("{} {} {} {}\n".format(
                t, min(v[1]), max(v[1]), v[0]))


    if paint_sphere:
        spheres = {}
        radii = {30:35,
                 60:25,
                 100:15,
                 1000:11,
                 }
        radii = {30:71,
                 60:51,
                 100:31,
                 1000:21,
                 }

        for th, r in radii.items():
            sphere_shape = (max(3, r//voxel_size[1]+1), r, r)
            zh = sphere_shape[0]//2
            yh = sphere_shape[1]//2
            xh = sphere_shape[2]//2
            sphere_rad = (sphere_shape[0]/2,
                          sphere_shape[1]/2,
                          sphere_shape[2]/2)
            sphere = rg.ellipsoid(sphere_shape, sphere_rad)
            spheres[th] = [sphere, zh, yh, xh]
        # print(sphere)
        # print(shape, sphere.shape)
    for f in range(start_frame, end_frame+1):
        arr = np.zeros(shape[1:], dtype=np.uint16)
        for c, v in node_to_track.items():
            if f != v[2]:
                continue
            t = graph.nodes[c]['t']
            z = int(graph.nodes[c]['z']/voxel_size[1])
            y = int(graph.nodes[c]['y']/voxel_size[2])
            x = int(graph.nodes[c]['x']/voxel_size[3])
            print(v[0], f, t, z, y, x, c, v)
            if paint_sphere:
                if isinstance(spheres, dict):
                    for th in sorted(spheres.keys()):
                        if t < int(th):
                            sphere, zh, yh, xh = spheres[th]
                            break
                try:
                    arr[(z-zh):(z+zh+1),
                        (y-yh):(y+yh+1),
                        (x-xh):(x+xh+1)] = sphere * v[0]
                except ValueError as e:
                    print(e)
                    print(z, zh, y, yh, x, xh, sphere.shape)
                    print(z-zh, z+zh+1, y-yh, y+yh+1, x-xh, x+xh+1, sphere.shape)
                    sphereT = np.copy(sphere)
                    if z-zh < 0:
                        zz1 = 0
                        sphereT = sphereT[(-(z-zh)):,...]
                        print(sphereT.shape)
                    else:
                        zz1 = z-zh
                    if z+zh+1 >= arr.shape[0]:
                        zz2 = arr.shape[0]
                        zt = arr.shape[0] - (z+zh+1)
                        sphereT = sphereT[:zt,...]
                        print(sphereT.shape)
                    else:
                        zz2 = z+zh+1

                    if y-yh < 0:
                        yy1 = 0
                        sphereT = sphereT[:, (-(y - yh)) :, ...]
                        print(sphereT.shape)
                    else:
                        yy1 = y-yh
                    if y+yh+1 >= arr.shape[1]:
                        yy2 = arr.shape[1]
                        yt = arr.shape[1] - (y+yh+1)
                        sphereT = sphereT[:,:yt,...]
                        print(sphereT.shape)
                    else:
                        yy2 = y+yh+1

                    if x-xh < 0:
                        xx1 = 0
                        sphereT = sphereT[...,(-(x-xh)):]
                        print(sphereT.shape)
                    else:
                        xx1 = x-xh
                    if x+xh+1 >= arr.shape[2]:
                        xx2 = arr.shape[2]
                        xt = arr.shape[2] - (x+xh+1)
                        sphereT = sphereT[...,:xt]
                        print(sphereT.shape)
                    else:
                        xx2 = x+xh+1

                    print(zz1, zz2, yy1, yy2, xx1, xx2)
                    arr[zz1:zz2,
                        yy1:yy2,
                        xx1:xx2] = sphereT * v[0]
                    # raise e
            else:
                arr[z, y, x] = v[0]
        tifffile.imwrite(os.path.join(
            out_dir, tif_fn.format(f)), arr,
                         compress=3)
