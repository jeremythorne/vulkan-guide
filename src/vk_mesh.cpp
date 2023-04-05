#include <tiny_obj_loader.h>
#include "vk_mesh.h"
#include <iostream>

VertexInputDescription Vertex::get_vertex_description() {
    VertexInputDescription description;

    description.bindings = {
        {
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
        }
    };

    description.attributes = {
        {
        .location = 0,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(Vertex, position)
        },
        {
        .location = 1,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(Vertex, normal)
        },
        {
        .location = 2,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(Vertex, color)
        }
    };
    return description;
}

bool Mesh::load_from_obj(const char * filename) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    
    std::string warn;
    std::string err;
    
    tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename,
        nullptr);

    if (!warn.empty()) {
        std::cout << "WARN: " << warn << std::endl;
    }
    if (!err.empty()) {
        std::cerr << "ERROR: " << err << std::endl;
        return false;
    }

    for (size_t s = 0; s < shapes.size(); s++) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            int fv = 3;
            for (size_t v = 0; v < fv; v++) {
                tinyobj::index_t idx =
                    shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
                tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
                tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
                tinyobj::real_t nx = attrib.normals[3 * idx.normal_index + 0];
                tinyobj::real_t ny = attrib.normals[3 * idx.normal_index + 1];
                tinyobj::real_t nz = attrib.normals[3 * idx.normal_index + 2];

                Vertex new_vert{
                    .position = {vx, vy, vz},
                    .normal = {nx, ny, nz},
                    .color = {nx, ny, nz},
                };
                _vertices.push_back(new_vert);
            }
            index_offset += fv;
        }
    }
    return true;
}
