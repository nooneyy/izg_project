/*!
 * @file
 * @brief This file contains implementation of gpu
 *
 * @author Tomáš Milet, imilet@fit.vutbr.cz
 */

#include "solutionInterface/gpu.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

void process_clear_color(GPUMemory &mem, ClearColorCommand cmd);
void process_clear_depth(GPUMemory &mem, ClearDepthCommand cmd);
void process_clear_stencil(GPUMemory &mem, ClearStencilCommand cmd);
void process_user_command(GPUMemory &mem, UserCommand cmd);
void process_draw(GPUMemory &mem, DrawCommand cmd);
void process_subcommand(GPUMemory &mem, SubCommand cmd);

typedef struct {
  std::array<OutVertex, 3> vertices;
} Primitive;

//! [student_GPU_run]
void student_GPU_run(GPUMemory &mem, CommandBuffer const &cb) {
  /// \todo Tato funkce reprezentuje funkcionalitu grafické karty.<br>
  /// Měla by umět zpracovat command buffer, čistit framebuffer a kreslit.<br>
  /// mem obsahuje paměť grafické karty.
  /// cb obsahuje command buffer pro zpracování.
  /// Bližší informace jsou uvedeny na hlavní stránce dokumentace.
  ///
  /// V základu jde o to, že cb obsahuje příkazy, které se musí provést nad
  /// pamětí mem. Správně fungující grafická karta dobře interpretuje příkazy v
  /// cb a správně změní obsah paměti mem.

  mem.gl_DrawID = 0;

  for (const auto &cmd : cb.commands) {
    switch (cmd.type) {
    case CommandType::BIND_FRAMEBUFFER:
      mem.activatedFramebuffer = cmd.data.bindFramebufferCommand.id;
      break;
    case CommandType::BIND_PROGRAM:
      mem.activatedProgram = cmd.data.bindProgramCommand.id;
      break;
    case CommandType::BIND_VERTEXARRAY:
      mem.activatedVertexArray = cmd.data.bindVertexArrayCommand.id;
      break;
    case CommandType::BLOCK_WRITES_COMMAND:
      mem.blockWrites = cmd.data.blockWritesCommand.blockWrites;
      break;
    case CommandType::SET_BACKFACE_CULLING_COMMAND:
      mem.backfaceCulling.enabled = cmd.data.setBackfaceCullingCommand.enabled;
      break;
    case CommandType::SET_FRONT_FACE_COMMAND:
      mem.backfaceCulling.frontFaceIsCounterClockWise =
          cmd.data.setFrontFaceCommand.frontFaceIsCounterClockWise;
      break;
    case CommandType::SET_STENCIL_COMMAND:
      mem.stencilSettings = cmd.data.setStencilCommand.settings;
      break;
    case CommandType::SET_DRAW_ID:
      mem.gl_DrawID = cmd.data.setDrawIdCommand.id;
      break;
    case CommandType::CLEAR_COLOR:
      process_clear_color(mem, cmd.data.clearColorCommand);
      break;
    case CommandType::CLEAR_DEPTH:
      process_clear_depth(mem, cmd.data.clearDepthCommand);
      break;
    case CommandType::CLEAR_STENCIL:
      process_clear_stencil(mem, cmd.data.clearStencilCommand);
      break;
    case CommandType::USER_COMMAND:
      process_user_command(mem, cmd.data.userCommand);
      break;
    case CommandType::DRAW:
      process_draw(mem, cmd.data.drawCommand);
      break;
    case CommandType::SUB_COMMAND:
      process_subcommand(mem, cmd.data.subCommand);
    default:
      break;
    }
  }
}
//! [student_GPU_run]

void process_clear_color(GPUMemory &mem, ClearColorCommand cmd) {
  const auto &selected_framebuffer = mem.framebuffers[mem.activatedFramebuffer];

  const auto &color_image = selected_framebuffer.color;
  const auto image_data = static_cast<uint8_t *>(color_image.data);

  if (image_data) {
    const auto value_count = selected_framebuffer.width *
                             selected_framebuffer.height *
                             color_image.bytesPerPixel;
    const auto channel_count = color_image.channels;
    if (color_image.format == Image::U8) {
      const uint8_t transformed_values[] = {
          static_cast<uint8_t>(cmd.value.x * UINT8_MAX),
          static_cast<uint8_t>(cmd.value.y * UINT8_MAX),
          static_cast<uint8_t>(cmd.value.z * UINT8_MAX),
          static_cast<uint8_t>(cmd.value.w * UINT8_MAX)};
      for (unsigned idx = 0; idx < value_count; idx++) {
        image_data[idx] = transformed_values[idx % channel_count];
      }
    } else {
      for (unsigned idx = 0; idx < value_count; idx++) {
        ((float *)image_data)[idx] = cmd.value[idx % channel_count];
      }
    }
  };
}

void process_clear_depth(GPUMemory &mem, ClearDepthCommand cmd) {
  const auto &selected_framebuffer = mem.framebuffers[mem.activatedFramebuffer];

  const auto &depth_image = selected_framebuffer.depth;
  const auto image_data = static_cast<float *>(depth_image.data);

  if (image_data) {
    const auto value_count =
        selected_framebuffer.width * selected_framebuffer.height;
    for (unsigned idx = 0; idx < value_count; idx++)
      image_data[idx] = cmd.value;
  }
}

void process_clear_stencil(GPUMemory &mem, ClearStencilCommand cmd) {
  const auto &selected_framebuffer = mem.framebuffers[mem.activatedFramebuffer];

  const auto &stencil_image = selected_framebuffer.stencil;
  const auto image_data = static_cast<uint8_t *>(stencil_image.data);

  if (image_data) {
    const auto value_count =
        selected_framebuffer.width * selected_framebuffer.height;
    for (unsigned idx = 0; idx < value_count; idx++)
      image_data[idx] = cmd.value;
  }
}

void process_user_command(GPUMemory &mem, UserCommand cmd) {
  if (cmd.callback) {
    (cmd.callback)(cmd.data);
  }
}

uint32_t get_vertex_id(GPUMemory &mem, uint32_t invocation) {
  const auto &current_vertex_array = mem.vertexArrays[mem.activatedVertexArray];

  if (current_vertex_array.indexBufferID == -1)
    return invocation;

  const auto &buf = mem.buffers[current_vertex_array.indexBufferID];
  const auto buffer_bytes = static_cast<const uint8_t *>(buf.data);
  const auto index_size = static_cast<uint8_t>(current_vertex_array.indexType);

  uint32_t result = 0;
  memcpy(&result,
         buffer_bytes + current_vertex_array.indexOffset +
             invocation * index_size,
         index_size);
  return result;
}

template <typename T>
T set_vertex_attrib(const VertexAttrib &attrib, const Buffer &buf,
                    uint32_t gl_VertexID) {
  T result;
  const auto buffer_bytes = static_cast<const uint8_t *>(buf.data);
  memcpy(&result, buffer_bytes + attrib.offset + (attrib.stride * gl_VertexID),
         sizeof(T));
  return result;
}

void fill_vertex_attribs(GPUMemory &mem, uint32_t gl_VertexID,
                         Attrib (&attribs)[maxAttribs]) {
  const auto &current_vertex_array = mem.vertexArrays[mem.activatedVertexArray];

  for (unsigned attr_idx = 0; attr_idx < maxAttribs; attr_idx++) {
    const auto &attrib = current_vertex_array.vertexAttrib[attr_idx];

    if (attrib.type == AttribType::EMPTY)
      continue;
    const auto &buf = mem.buffers[attrib.bufferID];

    switch (attrib.type) {
    case AttribType::FLOAT:
      attribs[attr_idx].v1 = set_vertex_attrib<float>(attrib, buf, gl_VertexID);
      break;
    case AttribType::VEC2:
      attribs[attr_idx].v2 =
          set_vertex_attrib<glm::vec2>(attrib, buf, gl_VertexID);
      break;
    case AttribType::VEC3:
      attribs[attr_idx].v3 =
          set_vertex_attrib<glm::vec3>(attrib, buf, gl_VertexID);
      break;
    case AttribType::VEC4:
      attribs[attr_idx].v4 =
          set_vertex_attrib<glm::vec4>(attrib, buf, gl_VertexID);
      break;
    case AttribType::UINT:
      attribs[attr_idx].u1 =
          set_vertex_attrib<uint32_t>(attrib, buf, gl_VertexID);
      break;
    case AttribType::UVEC2:
      attribs[attr_idx].u2 =
          set_vertex_attrib<glm::uvec2>(attrib, buf, gl_VertexID);
      break;
    case AttribType::UVEC3:
      attribs[attr_idx].u3 =
          set_vertex_attrib<glm::uvec3>(attrib, buf, gl_VertexID);
      break;
    case AttribType::UVEC4:
      attribs[attr_idx].u4 =
          set_vertex_attrib<glm::uvec4>(attrib, buf, gl_VertexID);
      break;
    default:
      break;
    }
  }
}

void call_vertex_shader(GPUMemory &mem, uint32_t invocation,
                        ShaderInterface &shader, OutVertex &out) {
  const auto &current_program = mem.programs[mem.activatedProgram];
  const auto vertex_shader = current_program.vertexShader;

  if (vertex_shader) {
    const auto gl_VertexID = get_vertex_id(mem, invocation);
    InVertex in = {.attributes = {}, .gl_VertexID = gl_VertexID};
    fill_vertex_attribs(mem, gl_VertexID, in.attributes);

    vertex_shader(out, in, shader);
  }
}

Primitive primitive_assembly(GPUMemory &mem, uint32_t vertex_idx,
                             ShaderInterface &shader) {
  Primitive primitive;
  for (unsigned arr_pos = 0; arr_pos < primitive.vertices.size(); arr_pos++)
    call_vertex_shader(mem, vertex_idx + arr_pos, shader,
                       primitive.vertices[arr_pos]);
  return primitive;
}

void perspective_div(Primitive &prim) {
  for (auto &vert : prim.vertices) {
    const auto w = vert.gl_Position.w;
    vert.gl_Position.x /= w;
    vert.gl_Position.y /= w;
    vert.gl_Position.z /= w;
  }
}

void viewport_transform(GPUMemory &mem, Primitive &prim) {
  const auto &selected_framebuffer = mem.framebuffers[mem.activatedFramebuffer];

  const auto width = selected_framebuffer.width;
  const auto height = selected_framebuffer.height;

  for (auto &vert : prim.vertices) {
    vert.gl_Position.x = (vert.gl_Position.x * 0.5 + 0.5) * width;
    vert.gl_Position.y = (vert.gl_Position.y * 0.5 + 0.5) * height;
  }
}

float edge_function(glm::vec4 const &from, glm::vec4 const &to,
                    glm::vec2 const &point) {
  return (point.x - from.x) * (to.y - from.y) -
         (point.y - from.y) * (to.x - from.x);
}

void interpolate_frag_attribs(InFragment &frag,
                              const AttribType (&vs2fs)[maxAttribs],
                              const glm::vec3 &lambda, Primitive &prim) {
  auto v0 = prim.vertices[0];
  auto v1 = prim.vertices[1];
  auto v2 = prim.vertices[2];

  auto persp_lambda_x = lambda.x / v0.gl_Position.w;
  auto persp_lambda_y = lambda.y / v1.gl_Position.w;
  auto persp_lambda_z = lambda.z / v2.gl_Position.w;

  const auto divisor = persp_lambda_x + persp_lambda_y + persp_lambda_z;

  persp_lambda_x /= divisor;
  persp_lambda_y /= divisor;
  persp_lambda_z /= divisor;

  for (unsigned attr_idx = 0; attr_idx < maxAttribs; attr_idx++) {
    switch (vs2fs[attr_idx]) {
    case AttribType::EMPTY:
      break;
    case AttribType::UINT:
    case AttribType::UVEC2:
    case AttribType::UVEC3:
    case AttribType::UVEC4:
      frag.attributes[attr_idx] = v0.attributes[attr_idx];
      break;
    case AttribType::FLOAT:
      frag.attributes[attr_idx].v1 =
          (persp_lambda_x * v0.attributes[attr_idx].v1) +
          (persp_lambda_y * v1.attributes[attr_idx].v1) +
          (persp_lambda_z * v2.attributes[attr_idx].v1);
      break;
    case AttribType::VEC2:
      frag.attributes[attr_idx].v2 =
          (persp_lambda_x * v0.attributes[attr_idx].v2) +
          (persp_lambda_y * v1.attributes[attr_idx].v2) +
          (persp_lambda_z * v2.attributes[attr_idx].v2);
      break;
    case AttribType::VEC3:
      frag.attributes[attr_idx].v3 =
          (persp_lambda_x * v0.attributes[attr_idx].v3) +
          (persp_lambda_y * v1.attributes[attr_idx].v3) +
          (persp_lambda_z * v2.attributes[attr_idx].v3);
      break;
    case AttribType::VEC4:
      frag.attributes[attr_idx].v4 =
          (persp_lambda_x * v0.attributes[attr_idx].v4) +
          (persp_lambda_y * v1.attributes[attr_idx].v4) +
          (persp_lambda_z * v2.attributes[attr_idx].v4);
      break;
    default:
      break;
    }
  }
}

void rasterize(GPUMemory &mem, Primitive &prim) {
  const auto &selected_framebuffer = mem.framebuffers[mem.activatedFramebuffer];
  const auto &current_program = mem.programs[mem.activatedProgram];
  const auto fragment_shader = current_program.fragmentShader;

  const auto width = selected_framebuffer.width;
  const auto height = selected_framebuffer.height;

  if (!fragment_shader || width == 0 || height == 0)
    return;

  const auto &v0_pos = prim.vertices[0].gl_Position;
  const auto &v1_pos = prim.vertices[1].gl_Position;
  const auto &v2_pos = prim.vertices[2].gl_Position;

  const auto [min_x, max_x] = std::minmax({v0_pos.x, v1_pos.x, v2_pos.x});
  const auto [min_y, max_y] = std::minmax({v0_pos.y, v1_pos.y, v2_pos.y});

  const auto min_x_px = std::max(0, static_cast<int>(std::floor(min_x)));
  const auto max_x_px =
      std::min(static_cast<int>(width) - 1, static_cast<int>(std::ceil(max_x)));

  const auto min_y_px = std::max(0, static_cast<int>(std::floor(min_y)));
  const auto max_y_px = std::min(static_cast<int>(height) - 1,
                                 static_cast<int>(std::ceil(max_y)));

  const auto edge0_step_x = v2_pos.y - v1_pos.y;
  const auto edge1_step_x = v0_pos.y - v2_pos.y;
  const auto edge2_step_x = v1_pos.y - v0_pos.y;

  const auto edge0_step_y = v1_pos.x - v2_pos.x;
  const auto edge1_step_y = v2_pos.x - v0_pos.x;
  const auto edge2_step_y = v0_pos.x - v1_pos.x;

  const auto area = edge_function(v0_pos, v1_pos, glm::vec2(v2_pos));
  if (area == 0.0f)
    return;

  if (mem.backfaceCulling.enabled &&
      (area < 0.0f) != mem.backfaceCulling.frontFaceIsCounterClockWise)
    return;

  const glm::vec2 first_sample{static_cast<float>(min_x_px) + 0.5f,
                               static_cast<float>(min_y_px) + 0.5f};

  auto row_edge0 = edge_function(v1_pos, v2_pos, first_sample);
  auto row_edge1 = edge_function(v2_pos, v0_pos, first_sample);
  auto row_edge2 = edge_function(v0_pos, v1_pos, first_sample);

  ShaderInterface shader = {.uniforms = mem.uniforms,
                            .textures = mem.textures,
                            .gl_DrawID = mem.gl_DrawID};

  for (int y = min_y_px; y <= max_y_px; y++) {
    auto pixel_edge0 = row_edge0;
    auto pixel_edge1 = row_edge1;
    auto pixel_edge2 = row_edge2;

    for (int x = min_x_px; x <= max_x_px; x++) {
      const auto is_inside =
          (pixel_edge0 >= 0.0f && pixel_edge1 >= 0.0f && pixel_edge2 >= 0.0f) ||
          (pixel_edge0 <= 0.0f && pixel_edge1 <= 0.0f && pixel_edge2 <= 0.0f);

      if (is_inside) {
        const glm::vec3 lambda = {pixel_edge0 / area, pixel_edge1 / area,
                                  pixel_edge2 / area};

        InFragment in_fragment = {};
        in_fragment.gl_FragCoord = {
            static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f,
            lambda.x * v0_pos.z + lambda.y * v1_pos.z + lambda.z * v2_pos.z,
            1.0f};
        interpolate_frag_attribs(in_fragment, current_program.vs2fs, lambda,
                                 prim);

        OutFragment out_fragment = {};
        fragment_shader(out_fragment, in_fragment, shader);
      }

      pixel_edge0 += edge0_step_x;
      pixel_edge1 += edge1_step_x;
      pixel_edge2 += edge2_step_x;
    }

    row_edge0 += edge0_step_y;
    row_edge1 += edge1_step_y;
    row_edge2 += edge2_step_y;
  }
};

void process_draw(GPUMemory &mem, DrawCommand cmd) {
  // todo
  ShaderInterface shader = {.uniforms = mem.uniforms,
                            .textures = mem.textures,
                            .gl_DrawID = mem.gl_DrawID};
  for (uint32_t vertex_idx = 0; vertex_idx < cmd.nofVertices; vertex_idx += 3) {
    Primitive primitive = primitive_assembly(mem, vertex_idx, shader);
    perspective_div(primitive);
    viewport_transform(mem, primitive);
    rasterize(mem, primitive);
  }
  mem.gl_DrawID++;
}

void process_subcommand(GPUMemory &mem, SubCommand cmd) {
  if (cmd.commandBuffer) {
    student_GPU_run(mem, *cmd.commandBuffer);
  }
}