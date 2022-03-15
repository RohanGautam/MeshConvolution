//
// Created by zhouyi on 6/14/19.
//

#ifndef POINTSAMPLING_MESHCNN_H
#define POINTSAMPLING_MESHCNN_H

#endif //POINTSAMPLING_MESHCNN_H

#include "meshPooler.h"
#include "cnpy/cnpy.h"


//gives you an fully convolutional auto-encoder. The pooling and unpooling strides are symmetric.
class MeshCNN
{
public:
    Mesh _mesh;
    vector<MeshPooler> _meshPoolers;

    MeshCNN(Mesh mesh) {_mesh = mesh;}

    //stride==1 means no pool
    void add_pool_layer(int stride, int radius_pool, int radius_unpool, bool use_full_search_radius_for_center_to_center_map=true)
    {
        cout<<"## Add Pooling "<<_meshPoolers.size()<<"\n";//<<". stride: "<<stride<<" radius_pool: "<<radius_pool<<" radius_unpool: "<<radius_unpool<<"\n";
        if(_meshPoolers.size()==0)
        {
            MeshPooler meshPooler = MeshPooler(use_full_search_radius_for_center_to_center_map);
            meshPooler.set_connection_map_from_mesh(_mesh);
            meshPooler.set_must_include_center_lst_from_mesh(_mesh);
            meshPooler.compute_pool_and_unpool_map(stride, radius_pool, radius_unpool);
            _meshPoolers.push_back(meshPooler);
            return;
        }

        MeshPooler meshPooler=MeshPooler(use_full_search_radius_for_center_to_center_map);
        meshPooler._connection_map = _meshPoolers.back()._center_center_map; //set the last pooler's center_center_map as the connection map of the current one.

        meshPooler._must_include_center_lst=_meshPoolers.back()._must_include_center_lst_new_index;
        //cout<<"must include centers num: "<<meshPooler._must_include_center_lst.size()<<"\n";
        meshPooler.compute_pool_and_unpool_map(stride, radius_pool,radius_unpool);
        _meshPoolers.push_back(meshPooler);
        return;

    }


    //neighbour_lst_lst  current_size*(1+max_neighbor_num*2).
    //               neighbor_num, (neighbor_id0,dist0), (neighbor_id1,dist1) ..., (neighbor_idx,distx), (input_size,-1), ..., (input_size,-1)
    vector<int> get_neighborID_lst_lst(const vector<vector<Int2>> &pool_map, int previous_size)
    {
        int current_point_num = pool_map.size();

        //first count the maximum number of neighbors
        int max_neighbor_num = 0;
        for (int i=0;i<pool_map.size();i++)
        {
            int neighbor_num = pool_map[i].size();
            if(neighbor_num > max_neighbor_num)
                max_neighbor_num = neighbor_num;
        }

        vector<int> neighborID_lst_lst_flat;

        for (int i=0;i<pool_map.size();i++)
        {
            vector<int> neighborID_lst = vector<int>(max_neighbor_num*2);
            for (int j =0; j <pool_map[i].size(); j++)
            {
                neighborID_lst[j*2] = pool_map[i][j][0]; // id
                neighborID_lst[j*2+1] =pool_map[i][j][1];// dist

            }
            if(pool_map[i].size()<max_neighbor_num)
            {
                for (int j=pool_map[i].size(); j<max_neighbor_num;j++) {
                    neighborID_lst[j * 2] = previous_size;
                    neighborID_lst[j * 2+1] = -1;
                }
            }
            neighborID_lst_lst_flat.push_back(pool_map[i].size());

            neighborID_lst_lst_flat.insert(neighborID_lst_lst_flat.end(), neighborID_lst.begin(), neighborID_lst.end());

        }

        return neighborID_lst_lst_flat;
    }




    void save_pool_and_unpool_neighbor_info_to_npz(const string& save_path)
    {
        for(int i=0;i<_meshPoolers.size(); i++)
        {
            int after_pool_size = (_meshPoolers[i]._center_center_map.size());
            int before_pool_size = (_meshPoolers[i]._connection_map.size());

            vector<int> pool_neighborID_lst_lst = get_neighborID_lst_lst(_meshPoolers[i]._pool_map, before_pool_size);

            vector<int> unpool_neighborID_lst_lst = get_neighborID_lst_lst(_meshPoolers[i]._unpool_map, after_pool_size);


            cout<<"save "<<save_path+"_pool"+to_string(i)+".npy\n";
            int neighbor_num_pool_2 = pool_neighborID_lst_lst.size()/after_pool_size-1;
            
            std::vector<size_t > shape_info = {(size_t)after_pool_size, (size_t)(1+neighbor_num_pool_2) };
            cout<<"output point num: "<<shape_info[0]<<"; max_neighbor_num: "<<shape_info[1]<<"\n";

            // cnpy::npy_save(save_path+"_pool"+to_string(i)+".npy", &pool_neighborID_lst_lst [0], shape_info, "w");//"a" appends to the file we created above

            cout<<"save "<<save_path+"_unpool"+to_string(i)+".npy\n";
            int neighbor_num_unpool_2 = unpool_neighborID_lst_lst.size()/before_pool_size-1;
            shape_info = {(size_t)before_pool_size, (size_t)(1+neighbor_num_unpool_2) };
            cout<<"output point num: "<<shape_info[0]<<"; max_neighbor_num: "<<shape_info[1]<<"\n";
            // cnpy::npy_save(save_path+"_unpool"+to_string(i)+".npy", &unpool_neighborID_lst_lst [0], shape_info, "w");//"a" appends to the file we created above
        }


    }

};
