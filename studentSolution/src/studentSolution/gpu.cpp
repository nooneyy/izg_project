/*!
 * @file
 * @brief This file contains implementation of gpu
 *
 * @author Tomáš Milet, imilet@fit.vutbr.cz
 */

#include "solutionInterface/gpu.hpp"
#include <cstdint>
#include <studentSolution/gpu.hpp>

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

  for (auto cmd : cb.commands) {
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
  const auto selected_framebuffer = &mem.framebuffers[mem.activatedFramebuffer];

  const auto color_image = selected_framebuffer->color;
  const auto image_data = (uint8_t *)color_image.data;

  if (image_data) {
    const auto value_count = selected_framebuffer->width *
                             selected_framebuffer->height *
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
  const auto selected_framebuffer = &mem.framebuffers[mem.activatedFramebuffer];

  const auto depth_image = selected_framebuffer->depth;
  const auto image_data = (float *)depth_image.data;

  if (image_data) {
    const auto value_count =
        selected_framebuffer->width * selected_framebuffer->height;
    for (unsigned idx = 0; idx < value_count; idx++)
      image_data[idx] = cmd.value;
  }
}

void process_clear_stencil(GPUMemory &mem, ClearStencilCommand cmd) {
  const auto selected_framebuffer = &mem.framebuffers[mem.activatedFramebuffer];

  const auto depth_image = selected_framebuffer->stencil;
  const auto image_data = (uint8_t *)depth_image.data;

  if (image_data) {
    const auto value_count =
        selected_framebuffer->width * selected_framebuffer->height;
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
  const auto current_vertex_array = mem.vertexArrays[mem.activatedVertexArray];

  if (current_vertex_array.indexBufferID == -1)
    return invocation;

  const auto buf = mem.buffers[current_vertex_array.indexBufferID];
  const auto buf_idx = invocation + current_vertex_array.indexOffset;
  switch (current_vertex_array.indexType) {
  case IndexType::U8:
    return ((uint8_t *)buf.data)[buf_idx];
    break;
  case IndexType::U16:
    return ((uint16_t *)buf.data)[buf_idx];
    break;
  case IndexType::U32:
  default:
    return ((uint32_t *)buf.data)[buf_idx];
    break;
  }
}

// void get_vertex_attribs(GPUMemory &mem, uint32_t gl_VertexId) {
//   Attrib attribs[maxAttribs];
//   const auto current_vertex_array =
//   mem.vertexArrays[mem.activatedVertexArray];

//   for (auto attrib : current_vertex_array.vertexAttrib) {
//     if (attrib.type != AttribType::EMPTY) {
//       auto buf = mem.buffers[attrib.bufferID];
//       memcpy(attribs[0],
//              ((void *)buf.data)[attrib.offset + gl_VertexId * attrib.stride],
//              attrib.type);
//     }
//   }
// }

void call_vertex_shader(GPUMemory &mem, uint32_t invocation,
                        ShaderInterface &shader) {
  const auto current_program = mem.programs[mem.activatedProgram];
  const auto vertex_shader = current_program.vertexShader;

  if (vertex_shader) {
    InVertex in = {.gl_VertexID = get_vertex_id(mem, invocation)};
    OutVertex out = {};
    vertex_shader(out, in, shader);
  }
}

void process_draw(GPUMemory &mem, DrawCommand cmd) {
  // todo
  ShaderInterface shader = {.uniforms = mem.uniforms,
                            .textures = mem.textures,
                            .gl_DrawID = mem.gl_DrawID};
  for (uint32_t invocation = 0; invocation < cmd.nofVertices; invocation++)
    call_vertex_shader(mem, invocation, shader);
  mem.gl_DrawID++;
}

void process_subcommand(GPUMemory &mem, SubCommand cmd) {
  if (cmd.commandBuffer) {
    student_GPU_run(mem, *cmd.commandBuffer);
  }
}