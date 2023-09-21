#include <iostream>
#include <cstdlib>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include "common.h"
#include "c_api.h"
#include "tengine_operations.h"

struct options opt;
float input_scale = 0.f;
int input_zero_point = 0;
const float mean[3] = {0, 0, 0};
const float scale[3] = {0.003921, 0.003921, 0.003921};
graph_t graph_head;

int main()
{
      std::string head_model = "/home/rpdzkj/Desktop/test_head/models/nanotrackv2_head1_uint8.tmfile";
      opt.num_thread = 1;
      opt.cluster = TENGINE_CLUSTER_ALL;
      opt.precision = TENGINE_MODE_UINT8;
      opt.affinity = 0;
      /* inital tengine */
      if (init_tengine() != 0)
      {
            fprintf(stderr, "Initial tengine failed.\n");
            return -1;
      }
      fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

      /* create VeriSilicon TIM-VX backend */
      context_t timvx_context = create_context("timvx", 1);
      int rtt = set_context_device(timvx_context, "TIMVX", nullptr, 0);
      if (0 > rtt)
      {
            fprintf(stderr, " add_context_device VSI DEVICE failed.\n");
            return -1;
      }

      graph_head = create_graph(timvx_context, "tengine", head_model.c_str());
      // graph_t graph_head = create_graph(NULL, "tengine", head_model.c_str());
      // if (graph_head == nullptr)
      // {
      //       fprintf(stderr, "Create graph failed.\n");
      //       return -1;
      // }

      cv::Mat image = cv::imread("/home/rpdzkj/Desktop/test_head/image/test_256_20.jpg");
      cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
      // char* image_file = "/home/rpdzkj/Desktop/test_head/image/test2.jpg";
      int image_size = 256 * 20 * 3;
      int dims[] = {1, 3, 256, 20};
      std::vector<uint8_t> input_data(image_size);

      tensor_t input_tensor = get_graph_input_tensor(graph_head, 0, 0);
      if (input_tensor == nullptr)
      {
            fprintf(stderr, "Get input tensor failed\n");
            return -1;
      }
      if (set_tensor_shape(input_tensor, dims, 4) < 0)
      {
            fprintf(stderr, "Set input tensor shape failed\n");
            return -1;
      }
      // 具体来说，它首先调用 set_tensor_buffer 函数，该函数接受三个参数：要设置的张量、数据缓冲区的指针和缓冲区大小。
      // 通过调用该函数，可以将输入数据 input_data 的内容复制到指定的 input_tensor 张量中。
      if (set_tensor_buffer(input_tensor, input_data.data(), image_size * sizeof(uint8_t)) < 0)
      {
            fprintf(stderr, "Set input tensor buffer failed\n");
            return -1;
      }

      /* prerun graph, set work options(num_thread, cluster, precision) */
      if (prerun_graph_multithread(graph_head, opt) < 0)
      {
            fprintf(stderr, "Prerun multithread graph failed.\n");
            return -1;
      }
      /* prepare process input data, set the data mem to input tensor */
      get_tensor_quant_param(input_tensor, &input_scale, &input_zero_point, 1);
      // nhwc -> nchw
      for (int h = 0; h < image.rows; h++)
      {
            for (int w = 0; w < image.cols; w++)
            {
                  for (int c = 0; c < 3; c++)
                  {
                        int in_index = h * image.cols * 3 + w * 3 + c;
                        int out_index = c * image.rows * image.cols + h * image.cols + w;
                        float input_fp32 = (image.data[in_index] - mean[c]) * scale[c];

                        /* quant to uint8 */
                        int udata = (round)(input_fp32 / input_scale + (float)input_zero_point);
                        if (udata > 255)
                              udata = 255;
                        else if (udata < 0)
                              udata = 0;

                        input_data[out_index] = udata;
                  }
            }
      }

      // get_input_data(image_file, input_data, 256, 40, mean, scale);
      if (run_graph(graph_head, 1) < 0)
      {
            fprintf(stderr, "Run graph failed\n");
            return -1;
      }

      tensor_t cls_output = get_graph_output_tensor(graph_head, 0, 0);
      tensor_t loc_output = get_graph_output_tensor(graph_head, 1, 0);

      int cls_count = get_tensor_buffer_size(cls_output) / sizeof(float);
      int loc_count = get_tensor_buffer_size(loc_output) / sizeof(float);
      std::vector<float> cls_data(cls_count, 0);
      std::vector<float> loc_data(loc_count, 0);
      float *cls_data_u8 = (float *)get_tensor_buffer(cls_output);
      float *loc_data_u8 = (float *)get_tensor_buffer(loc_output);

      // std::cout << sizeof(cls_data_u8) << std::endl;
      for (int c = 0; c < cls_count; c++)
      {
            cls_data[c] = (float)(*(cls_data_u8 + c));
      }

      for (int c = 0; c < loc_count; c++)
      {
            loc_data[c] = (float)(*(loc_data_u8 + c));
      }

      postrun_graph(graph_head);
      destroy_graph(graph_head);
      destroy_context(timvx_context);
      release_tengine();
}