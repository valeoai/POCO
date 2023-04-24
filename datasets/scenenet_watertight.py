import os
import subprocess
import open3d as o3d

# processed using the code from https://github.com/hjwdzh/Manifold

raw_data_dir="../downloadscenenet"
manifold_code_dir="./"

filenames =[
"1Bathroom/107_labels.obj",
"1Bathroom/1_labels.obj",
"1Bathroom/28_labels.obj",
"1Bathroom/29_labels.obj",
"1Bathroom/4_labels.obj",
"1Bathroom/5_labels.obj",
"1Bathroom/69_labels.obj",
"1Bedroom/3_labels.obj",
"1Bedroom/77_labels.obj",
"1Bedroom/bedroom27.obj",
"1Bedroom/bedroom_1.obj",
"1Bedroom/bedroom_68.obj",
"1Bedroom/bedroom_wenfagx.obj",
"1Bedroom/bedroom_xpg.obj",
"1Kitchen/1-14_labels.obj",
"1Kitchen/102.obj",
"1Kitchen/13_labels.obj",
"1Kitchen/2.obj",
"1Kitchen/35_labels.obj",
"1Kitchen/kitchen_106_blender_name_and_mat.obj",
"1Kitchen/kitchen_16_blender_name_and_mat.obj",
"1Kitchen/kitchen_76_blender_name_and_mat.obj",
"1Living-room/cnh_blender_name_and_mat.obj",
"1Living-room/living_room_33.obj",
"1Living-room/lr_kt7_blender_scene.obj",
"1Living-room/pg_blender_name_and_mat.obj",
"1Living-room/room_89_blender.obj",
"1Living-room/room_89_blender_no_paintings.obj",
"1Living-room/yoa_blender_name_mat.obj",
"1Office/2_crazy3dfree_labels.obj",
"1Office/2_hereisfree_labels.obj",
"1Office/4_3dmodel777.obj",
"1Office/4_hereisfree_labels.obj",
"1Office/7_crazy3dfree_labels.obj",
]

filenames = [os.path.join(raw_data_dir, filename) for filename in filenames]

for filename in filenames:
    print(filename)
    fname = filename.split("/")
    fname = fname[-2:]
    os.makedirs(fname[0], exist_ok=True)
    fname = os.path.join(fname[0], fname[1])

    # watertight
    subprocess.call([os.path.join(manifold_code_dir,"Manifold/build/manifold"), 
                    filename, "tmp.obj", "500000"])

    # mesh clean
    mesh = o3d.io.read_triangle_mesh("tmp.obj")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()

    o3d.io.write_triangle_mesh(fname+".ply", mesh)
