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
#include <cstdint>
#include <cstring>

void process_clear_color(GPUMemory &mem, ClearColorCommand cmd);
void process_clear_depth(GPUMemory &mem, ClearDepthCommand cmd);
void process_clear_stencil(GPUMemory &mem, ClearStencilCommand cmd);
void process_user_command(GPUMemory &mem, UserCommand cmd);
void process_draw(GPUMemory &mem, DrawCommand cmd);
void process_subcommand(GPUMemory &mem, SubCommand cmd);

struct Primitive {
  std::array<OutVertex, 3> vertices;
};

struct ClippedPrimitives {
  std::array<Primitive, 2> primitives;
  uint32_t count = 0;
};

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
    case CommandType::SET_BLENDING_COMMAND:
      mem.blendingSettings = cmd.data.setBlendingCommand.settings;
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

  auto &color_image = mem.framebuffers[mem.activatedFramebuffer].color;
  if (!color_image.data)
    return;

  const uint8_t transformed_values[] = {
      static_cast<uint8_t>(cmd.value.x * UINT8_MAX),
      static_cast<uint8_t>(cmd.value.y * UINT8_MAX),
      static_cast<uint8_t>(cmd.value.z * UINT8_MAX),
      static_cast<uint8_t>(cmd.value.w * UINT8_MAX)};

  for (uint32_t y = 0; y < selected_framebuffer.height; y++) {
    for (uint32_t x = 0; x < selected_framebuffer.width; x++) {
      auto *value = static_cast<uint8_t *>(getPixel(color_image, x, y));

      for (uint32_t channel = 0; channel < color_image.channels; channel++) {
        const auto rgba = color_image.channelTypes[channel];
        if (color_image.format == Image::U8) {
          value[channel] = transformed_values[rgba];
        } else {
          reinterpret_cast<float *>(value)[channel] = cmd.value[rgba];
        }
      }
    }
  }
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

float near_plane_distance(OutVertex const &vertex) {
  return vertex.gl_Position.z + vertex.gl_Position.w;
}

OutVertex interpolate_vertex_attribs(OutVertex const &vertex_a,
                                     OutVertex const &vertex_b, float t,
                                     AttribType const (&vs2fs)[maxAttribs]) {
  OutVertex result = {};
  result.gl_Position =
      vertex_a.gl_Position + t * (vertex_b.gl_Position - vertex_a.gl_Position);

  for (unsigned attr_idx = 0; attr_idx < maxAttribs; attr_idx++) {
    switch (vs2fs[attr_idx]) {
    case AttribType::FLOAT:
      result.attributes[attr_idx].v1 = vertex_a.attributes[attr_idx].v1 +
                                       t * (vertex_b.attributes[attr_idx].v1 -
                                            vertex_a.attributes[attr_idx].v1);
      break;
    case AttribType::VEC2:
      result.attributes[attr_idx].v2 = vertex_a.attributes[attr_idx].v2 +
                                       t * (vertex_b.attributes[attr_idx].v2 -
                                            vertex_a.attributes[attr_idx].v2);
      break;
    case AttribType::VEC3:
      result.attributes[attr_idx].v3 = vertex_a.attributes[attr_idx].v3 +
                                       t * (vertex_b.attributes[attr_idx].v3 -
                                            vertex_a.attributes[attr_idx].v3);
      break;
    case AttribType::VEC4:
      result.attributes[attr_idx].v4 = vertex_a.attributes[attr_idx].v4 +
                                       t * (vertex_b.attributes[attr_idx].v4 -
                                            vertex_a.attributes[attr_idx].v4);
      break;
    case AttribType::EMPTY:
    case AttribType::UINT:
    case AttribType::UVEC2:
    case AttribType::UVEC3:
    case AttribType::UVEC4:
    default:
      break;
    }
  }

  return result;
}

OutVertex make_intersection_vertex(OutVertex const &vertex_a,
                                   OutVertex const &vertex_b,
                                   AttribType const (&vs2fs)[maxAttribs]) {
  const auto &a_pos = vertex_a.gl_Position;
  const auto &b_pos = vertex_b.gl_Position;
  const auto t = (-a_pos.w - a_pos.z) / (b_pos.w - a_pos.w + b_pos.z - a_pos.z);
  return interpolate_vertex_attribs(vertex_a, vertex_b, t, vs2fs);
}

ClippedPrimitives near_plane_clipping(GPUMemory &mem, Primitive const &prim) {
  ClippedPrimitives clipped = {};
  const auto &vs2fs = mem.programs[mem.activatedProgram].vs2fs;
  std::array<OutVertex, 4> new_vertices;
  uint32_t vertex_count = 0;

  for (unsigned vertex_idx = 0; vertex_idx < 3; vertex_idx++) {
    const auto &vertex_a = prim.vertices[vertex_idx];
    const auto &vertex_b = prim.vertices[(vertex_idx + 2) % 3];
    const auto a_inside = near_plane_distance(vertex_a) > 0;
    const auto b_inside = near_plane_distance(vertex_b) > 0;

    if (a_inside && !b_inside)
      new_vertices[vertex_count++] =
          make_intersection_vertex(vertex_b, vertex_a, vs2fs);

    if (a_inside)
      new_vertices[vertex_count++] = vertex_a;

    if (!a_inside && b_inside)
      new_vertices[vertex_count++] =
          make_intersection_vertex(vertex_b, vertex_a, vs2fs);
  }

  if (vertex_count < 3)
    return clipped;

  clipped.primitives[0].vertices = {new_vertices[0], new_vertices[1],
                                    new_vertices[2]};
  clipped.count = 1;

  if (vertex_count == 4) {
    clipped.primitives[1].vertices = {new_vertices[0], new_vertices[2],
                                      new_vertices[3]};
    clipped.count = 2;
  }

  return clipped;
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
    float y = vert.gl_Position.y * 0.5 + 0.5;
    vert.gl_Position.y =
        (selected_framebuffer.yReversed ? (1 - y) : y) * height;
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
  auto &v0 = prim.vertices[0];
  auto &v1 = prim.vertices[1];
  auto &v2 = prim.vertices[2];

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

bool stencil_func_ret(StencilFunc func, uint8_t value, uint32_t ref_value) {
  switch (func) {
  case StencilFunc::NEVER:
    return false;
  case StencilFunc::LESS:
    return value < ref_value;
  case StencilFunc::LEQUAL:
    return value <= ref_value;
  case StencilFunc::GREATER:
    return value > ref_value;
  case StencilFunc::GEQUAL:
    return value >= ref_value;
  case StencilFunc::EQUAL:
    return value == ref_value;
  case StencilFunc::NOTEQUAL:
    return value != ref_value;
  case StencilFunc::ALWAYS:
  default:
    return true;
  }
}

uint8_t get_new_stencil_value(StencilOp op, uint8_t value, uint32_t ref_value) {
  switch (op) {
  case StencilOp::KEEP:
    return value;
  case StencilOp::ZERO:
    return 0;
  case StencilOp::REPLACE:
    return static_cast<uint8_t>(ref_value);
  case StencilOp::INCR:
    return value == UINT8_MAX ? value : value + 1;
  case StencilOp::INCR_WRAP:
    return value + 1;
  case StencilOp::DECR:
    return value == 0 ? value : value - 1;
  case StencilOp::DECR_WRAP:
    return value - 1;
  case StencilOp::INVERT:
    return ~value;
  default:
    return value;
  }
}

bool stencil_test(GPUMemory &mem, InFragment &frag, bool front_facing) {
  const auto stencil_settings = mem.stencilSettings;
  auto &stencil_buffer = mem.framebuffers[mem.activatedFramebuffer].stencil;

  if (!stencil_settings.enabled || !stencil_buffer.data)
    return true;

  const auto px_x = frag.gl_FragCoord.x;
  const auto px_y = frag.gl_FragCoord.y;

  auto value = static_cast<uint8_t *>(getPixel(stencil_buffer, px_x, px_y));

  if (stencil_func_ret(stencil_settings.func, *value,
                       stencil_settings.refValue))
    return true;

  if (!mem.blockWrites.stencil) {
    const auto sfail = front_facing ? stencil_settings.frontOps.sfail
                                    : stencil_settings.backOps.sfail;

    *value = get_new_stencil_value(sfail, *value, stencil_settings.refValue);
  }

  return false;
}

bool depth_test(GPUMemory &mem, InFragment &frag, bool front_facing) {
  const auto stencil_settings = mem.stencilSettings;
  const auto &depth_buffer = mem.framebuffers[mem.activatedFramebuffer].depth;

  if (!depth_buffer.data)
    return true;

  const auto px_x = frag.gl_FragCoord.x;
  const auto px_y = frag.gl_FragCoord.y;
  const auto pixel_depth = frag.gl_FragCoord.z;

  const auto depth_value =
      *static_cast<const float *>(getPixel(depth_buffer, px_x, px_y));

  if (depth_value > pixel_depth)
    return true;

  if (!mem.blockWrites.stencil && stencil_settings.enabled) {
    auto &stencil_buffer = mem.framebuffers[mem.activatedFramebuffer].stencil;

    if (!stencil_buffer.data)
      return false;

    auto stencil_value =
        static_cast<uint8_t *>(getPixel(stencil_buffer, px_x, px_y));

    const auto dpfail = front_facing ? stencil_settings.frontOps.dpfail
                                     : stencil_settings.backOps.dpfail;
    *stencil_value = get_new_stencil_value(dpfail, *stencil_value,
                                           stencil_settings.refValue);
  };
  return false;
}

glm::vec4 read_color_buffer(Image &color_buffer, uint32_t x, uint32_t y) {
  glm::vec4 color = {};
  auto *value = static_cast<uint8_t *>(getPixel(color_buffer, x, y));

  if (color_buffer.format == Image::U8) {
    for (unsigned channel = 0; channel < color_buffer.channels; channel++) {
      const auto rgba = color_buffer.channelTypes[channel];
      color[rgba] = value[channel] / static_cast<float>(UINT8_MAX);
    }
  } else {
    for (unsigned channel = 0; channel < color_buffer.channels; channel++) {
      const auto rgba = color_buffer.channelTypes[channel];
      color[rgba] = reinterpret_cast<float *>(value)[channel];
    }
  }

  if (color_buffer.channels < 4)
    color.w = color.x;

  return color;
}

void write_color_buffer(Image &color_buffer, uint32_t x, uint32_t y,
                        glm::vec4 &color) {
  auto *value = static_cast<uint8_t *>(getPixel(color_buffer, x, y));

  for (unsigned channel = 0; channel < color_buffer.channels; channel++) {
    const auto rgba = color_buffer.channelTypes[channel];
    if (color_buffer.format == Image::U8) {
      value[channel] = std::clamp<float>(color[rgba], 0, 1) * UINT8_MAX;
    } else {
      reinterpret_cast<float *>(value)[channel] = color[rgba];
    }
  }
}

glm::vec4 blending_factor(BlendFunc factor, glm::vec4 &src,
                          glm::vec4 const &dst) {
  switch (factor) {
  case BlendFunc::ZERO:
    return glm::vec4(0);
  case BlendFunc::SRC_COLOR:
    return src;
  case BlendFunc::ONE_MINUS_SRC_COLOR:
    return glm::vec4(1) - src;
  case BlendFunc::DST_COLOR:
    return dst;
  case BlendFunc::ONE_MINUS_DST_COLOR:
    return glm::vec4(1) - dst;
  case BlendFunc::SRC_ALPHA:
    return glm::vec4(src.a);
  case BlendFunc::ONE_MINUS_SRC_ALPHA:
    return glm::vec4(1 - src.a);
  case BlendFunc::DST_ALPHA:
    return glm::vec4(dst.a);
  case BlendFunc::ONE_MINUS_DST_ALPHA:
    return glm::vec4(1 - dst.a);
  case BlendFunc::ONE:
  default:
    return glm::vec4(1);
  }
}

glm::vec4 color_blending(BlendingSettings &settings, glm::vec4 &src,
                         glm::vec4 const &dst) {
  const auto src_factor = src * blending_factor(settings.sFactor, src, dst);
  const auto dst_factor = dst * blending_factor(settings.dFactor, src, dst);

  switch (settings.equation) {
  case BlendEquation::SUBTRACT:
    return src_factor - dst_factor;
  case BlendEquation::REVERSE_SUBTRACT:
    return dst_factor - src_factor;
  case BlendEquation::MIN:
    return glm::min(src, dst);
  case BlendEquation::MAX:
    return glm::max(src, dst);
  case BlendEquation::ADD:
  default:
    return src_factor + dst_factor;
  }
}

void handle_color_buffer(GPUMemory &mem, float px_x, float px_y,
                         OutFragment &out_frag) {
  if (mem.blockWrites.color)
    return;

  auto &color_buffer = mem.framebuffers[mem.activatedFramebuffer].color;
  if (!color_buffer.data)
    return;

  auto color = out_frag.gl_FragColor;
  const auto dst = read_color_buffer(color_buffer, px_x, px_y);
  if (mem.blendingSettings.enabled)
    color = color_blending(mem.blendingSettings, color, dst);

  write_color_buffer(color_buffer, px_x, px_y, color);
}

void late_per_fragment(GPUMemory &mem, InFragment &in_frag,
                       OutFragment &out_frag, bool front_facing) {
  if (out_frag.discard)
    return;

  const auto px_x = in_frag.gl_FragCoord.x;
  const auto px_y = in_frag.gl_FragCoord.y;
  const auto px_depth = in_frag.gl_FragCoord.z;

  const auto stencil_settings = mem.stencilSettings;
  auto &stencil_buffer = mem.framebuffers[mem.activatedFramebuffer].stencil;

  if (stencil_settings.enabled && stencil_buffer.data &&
      !mem.blockWrites.stencil) {
    auto stencil_value =
        static_cast<uint8_t *>(getPixel(stencil_buffer, px_x, px_y));

    const auto dppass = front_facing ? stencil_settings.frontOps.dppass
                                     : stencil_settings.backOps.dppass;

    *stencil_value = get_new_stencil_value(dppass, *stencil_value,
                                           stencil_settings.refValue);
  }

  auto &depth_buffer = mem.framebuffers[mem.activatedFramebuffer].depth;

  if (depth_buffer.data && !mem.blockWrites.depth) {
    auto depth_value = static_cast<float *>(getPixel(depth_buffer, px_x, px_y));
    *depth_value = px_depth;
  }

  handle_color_buffer(mem, px_x, px_y, out_frag);
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

  const auto front_facing =
      (area < 0.0f) == mem.backfaceCulling.frontFaceIsCounterClockWise;

  if (mem.backfaceCulling.enabled && !front_facing)
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
          (pixel_edge0 > 0.0f && pixel_edge1 > 0.0f && pixel_edge2 > 0.0f) ||
          (pixel_edge0 < 0.0f && pixel_edge1 < 0.0f && pixel_edge2 < 0.0f);

      if (is_inside) {
        const glm::vec3 lambda = {pixel_edge0 / area, pixel_edge1 / area,
                                  pixel_edge2 / area};

        InFragment in_fragment = {};
        in_fragment.gl_FragCoord = {
            static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f,
            lambda.x * v0_pos.z + lambda.y * v1_pos.z + lambda.z * v2_pos.z,
            1.0f};

        if (stencil_test(mem, in_fragment, front_facing) &&
            depth_test(mem, in_fragment, front_facing)) {
          interpolate_frag_attribs(in_fragment, current_program.vs2fs, lambda,
                                   prim);

          OutFragment out_fragment = {};
          fragment_shader(out_fragment, in_fragment, shader);
          late_per_fragment(mem, in_fragment, out_fragment, front_facing);
        }
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
  ShaderInterface shader = {.uniforms = mem.uniforms,
                            .textures = mem.textures,
                            .gl_DrawID = mem.gl_DrawID};
  for (uint32_t vertex_idx = 0; vertex_idx < cmd.nofVertices; vertex_idx += 3) {
    Primitive primitive = primitive_assembly(mem, vertex_idx, shader);
    auto clipped = near_plane_clipping(mem, primitive);

    for (uint32_t primitive_idx = 0; primitive_idx < clipped.count;
         primitive_idx++) {
      auto &clipped_primitive = clipped.primitives[primitive_idx];
      perspective_div(clipped_primitive);
      viewport_transform(mem, clipped_primitive);
      rasterize(mem, clipped_primitive);
    }
  }
  mem.gl_DrawID++;
}

void process_subcommand(GPUMemory &mem, SubCommand cmd) {
  if (cmd.commandBuffer) {
    student_GPU_run(mem, *cmd.commandBuffer);
  }
}