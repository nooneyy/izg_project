/*!
 * @file
 * @brief This file contains class that represents graphic card.
 *
 * @author Tomáš Milet, imilet@fit.vutbr.cz
 */
#pragma once

#include <solutionInterface/gpu.hpp>

/**
 * @brief function that executes work stored in command buffer on the gpu
 * memory. This function represents the functionality of GPU. It can render
 * stuff, it can clear framebuffer. It can process work stored in command buffer
 *
 * @param mem gpu memory
 * @param cb command buffer - packaged of work sent to the gpu
 */
void student_GPU_run(GPUMemory &mem, CommandBuffer const &cb);

void process_clear_color(GPUMemory &mem, ClearColorCommand cmd);
void process_clear_depth(GPUMemory &mem, ClearDepthCommand cmd);
void process_clear_stencil(GPUMemory &mem, ClearStencilCommand cmd);
void process_user_command(GPUMemory &mem, UserCommand cmd);
void process_draw(GPUMemory &mem, DrawCommand cmd);
void process_subcommand(GPUMemory &mem, SubCommand cmd);