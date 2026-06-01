/*!
 * @file
 * @brief This file contains functions for model rendering
 *
 * @author Tomáš Milet, imilet@fit.vutbr.cz
 */
#include "glm/ext/vector_float4.hpp"
#include "solutionInterface/gpu.hpp"
#include "solutionInterface/taskFunctions.hpp"
#include <solutionInterface/uniformLocations.hpp>
#include <studentSolution/gpu.hpp>
#include <studentSolution/prepareModel.hpp>

void student_prepareNode(GPUMemory &mem, CommandBuffer &command_buffer,
                         Node const &node, Model const &model,
                         unsigned &draw_counter,
                         glm::mat4 const &parent_matrix) {
  const auto multiplied_matrix = parent_matrix * node.modelMatrix;

  if (node.mesh >= 0) {
    const auto &mesh = model.meshes[node.mesh];

    VertexArray vertex_arr = {};
    vertex_arr.indexBufferID = mesh.indexBufferID;
    vertex_arr.indexOffset = mesh.indexOffset;
    vertex_arr.indexType = mesh.indexType;
    vertex_arr.vertexAttrib[0] = mesh.position;
    vertex_arr.vertexAttrib[1] = mesh.normal;
    vertex_arr.vertexAttrib[2] = mesh.texCoord;

    mem.vertexArrays[draw_counter] = vertex_arr;

    // doufam ze muzeme pouzivat tyhle funkce, ukradl jsem je z testu :P
    pushBindVertexArrayCommand(command_buffer, draw_counter);
    pushSetBackfaceCullingCommand(command_buffer, !mesh.doubleSided);
    pushDrawCommand(command_buffer, mesh.nofIndices);

    mem.uniforms[getUniformLocation(draw_counter, MODEL_MATRIX)].m4 =
        multiplied_matrix;
    mem.uniforms[getUniformLocation(draw_counter,
                                    INVERSE_TRANSPOSE_MODEL_MATRIX)]
        .m4 = glm::transpose(glm::inverse(multiplied_matrix));
    mem.uniforms[getUniformLocation(draw_counter, DIFFUSE_COLOR)].v4 =
        mesh.diffuseColor;
    mem.uniforms[getUniformLocation(draw_counter, TEXTURE_ID)].i1 =
        mesh.diffuseTexture;
    mem.uniforms[getUniformLocation(draw_counter, DOUBLE_SIDED)].v1 =
        mesh.doubleSided;

    draw_counter++;
  }

  for (unsigned child_idx = 0; child_idx < node.nofChildren; child_idx++) {
    student_prepareNode(mem, command_buffer, node.children[child_idx], model,
                        draw_counter, multiplied_matrix);
  }
}

/**
 * @brief This function prepares model into memory and creates command buffer
 *
 * @param mem gpu memory
 * @param commandBuffer command buffer
 * @param model model structure
 */
//! [drawModel]
void student_prepareModel(GPUMemory &mem, CommandBuffer &commandBuffer,
                          Model const &model) {
  /// \todo Tato funkce připraví command buffer pro model a nastaví správně
  /// pamět grafické karty.<br> Vaším úkolem je správně projít model a vložit
  /// vykreslovací příkazy do commandBufferu. Zároveň musíte vložit do paměti
  /// textury, buffery a uniformní proměnné, které buffer command buffer
  /// využívat. Bližší informace jsou uvedeny na hlavní stránce dokumentace a v
  /// testech.

  unsigned draw_counter = 0;
  glm::mat4 parent_matrix{1};

  for (unsigned buffer_idx = 0; buffer_idx < model.nofBuffers; buffer_idx++)
    mem.buffers[buffer_idx] = model.buffers[buffer_idx];

  for (unsigned texture_idx = 0; texture_idx < model.nofTextures; texture_idx++)
    mem.textures[texture_idx] = model.textures[texture_idx];

  for (unsigned root_idx = 0; root_idx < model.nofRoots; root_idx++) {
    student_prepareNode(mem, commandBuffer, model.roots[root_idx], model,
                        draw_counter, parent_matrix);
  }
}
//! [drawModel]

/**
 * @brief This function represents vertex shader of texture rendering method.
 *
 * @param outVertex output vertex
 * @param inVertex input vertex
 * @param si shader interface
 */
//! [drawModel_vs]
void student_drawModel_vertexShader(OutVertex &outVertex,
                                    InVertex const &inVertex,
                                    ShaderInterface const &si) {
  /// \todo Tato funkce reprezentujte vertex shader.<br>
  /// Vaším úkolem je správně trasnformovat vrcholy modelu.
  /// Bližší informace jsou uvedeny na hlavní stránce dokumentace.

  const auto &model_matrix =
      si.uniforms[getUniformLocation(si.gl_DrawID, MODEL_MATRIX)].m4;
  const auto &proj_view_matrix =
      si.uniforms[getUniformLocation(si.gl_DrawID, PROJECTION_VIEW_MATRIX)].m4;
  const auto &inv_transp_matrix =
      si.uniforms[getUniformLocation(si.gl_DrawID,
                                     INVERSE_TRANSPOSE_MODEL_MATRIX)]
          .m4;
  const auto &light_proj_matrix =
      si.uniforms[getUniformLocation(si.gl_DrawID, USE_SHADOW_MAP_MATRIX)].m4;

  const auto coords_vec4 = glm::vec4(inVertex.attributes[0].v3, 1);
  const auto normal_vec4 = glm::vec4(inVertex.attributes[1].v3, 0);

  const auto world_space_pos = model_matrix * coords_vec4;

  outVertex.gl_Position = proj_view_matrix * world_space_pos;

  outVertex.attributes[0].v3 = world_space_pos;
  outVertex.attributes[1].v3 = inv_transp_matrix * normal_vec4;
  outVertex.attributes[2].v2 = inVertex.attributes[2].v2;
  outVertex.attributes[3].v4 = light_proj_matrix * world_space_pos;
}
//! [drawModel_vs]

/**
 * @brief This functionrepresents fragment shader of texture rendering method.
 *
 * @param outFragment output fragment
 * @param inFragment input fragment
 * @param si shader interface
 */
//! [drawModel_fs]
void student_drawModel_fragmentShader(OutFragment &outFragment,
                                      InFragment const &inFragment,
                                      ShaderInterface const &si) {
  /// \todo Tato funkce reprezentujte fragment shader.<br>
  /// Vašim úkolem je správně obarvit fragmenty a osvětlit je pomocí lambertova
  /// osvětlovacího modelu. Bližší informace jsou uvedeny na hlavní stránce
  /// dokumentace.

  const auto &light_position =
      si.uniforms[getUniformLocation(si.gl_DrawID, LIGHT_POSITION)].v3;
  const auto &camera_position =
      si.uniforms[getUniformLocation(si.gl_DrawID, CAMERA_POSITION)].v3;
  const auto &shadow_map_id =
      si.uniforms[getUniformLocation(si.gl_DrawID, SHADOWMAP_ID)].i1;
  const auto &ambient_light_color =
      si.uniforms[getUniformLocation(si.gl_DrawID, AMBIENT_LIGHT_COLOR)].v3;
  const auto &light_color =
      si.uniforms[getUniformLocation(si.gl_DrawID, LIGHT_COLOR)].v3;
  const auto &diffuse_color =
      si.uniforms[getUniformLocation(si.gl_DrawID, DIFFUSE_COLOR)].v4;
  const auto &texture_id =
      si.uniforms[getUniformLocation(si.gl_DrawID, TEXTURE_ID)].i1;
  const auto &double_sided =
      si.uniforms[getUniformLocation(si.gl_DrawID, DOUBLE_SIDED)].v1;

  const auto &fragment_position = inFragment.attributes[0].v3;
  auto normalized_normal = glm::normalize(inFragment.attributes[1].v3);
  if (double_sided &&
      glm::dot(normalized_normal, fragment_position - camera_position) > 0.0f)
    normalized_normal *= -1;

  const auto real_diffuse_color =
      texture_id >= 0
          ? read_texture(si.textures[texture_id], inFragment.attributes[2].v2)
          : diffuse_color;
  const auto ambient_light =
      glm::vec3(real_diffuse_color) * ambient_light_color;

  const auto light_direction =
      glm::normalize(light_position - fragment_position);
  auto diffuse_factor =
      glm::clamp(glm::dot(light_direction, normalized_normal), 0.f, 1.f);

  if (shadow_map_id >= 0) {
    auto shadow_position = inFragment.attributes[3].v4;
    shadow_position /= shadow_position.w;

    if (shadow_position.x >= 0 && shadow_position.x <= 1 &&
        shadow_position.y >= 0 && shadow_position.y <= 1) {
      const auto depth =
          read_texture(si.textures[shadow_map_id], glm::vec2(shadow_position))
              .r;

      if (shadow_position.z > depth)
        diffuse_factor = 0;
    }
  }

  const auto diffused_light =
      glm::vec3(real_diffuse_color) * light_color * diffuse_factor;

  outFragment.gl_FragColor =
      glm::vec4(ambient_light + diffused_light, real_diffuse_color.a);
  outFragment.discard = real_diffuse_color.a < 0.5f;
}
//! [drawModel_fs]
