#include "mlp/activator.h"
#include <jsoncpp/json/json.h>
#include <stdio.h>
#include <fstream>
#include <map>
#include <string>

std::map<ACT_TYPE, const char*> act_type_map = {
    {type_ReLU, "ReLU"},
    {type_ReLU6, "ReLU6"},
    {type_Sigmoid, "Sigmoid"},
    {type_Tanh, "Tanh"},
    {type_Sigmoid_hard, "Sigmoid_Hard"},
    {type_Tanh_hard, "Tanh_Hard"},
    {type_Sign, "Sign"},
    {type_Softmax, "Softmax"},
    {type_LeakyReLU, "LeakyReLU"}
};

int main() {
    std::ifstream ifs("../python_scripts_helpers/activator_test.json");
    if (!ifs) {
        printf("activator_config.json not found\n");    
        return -1;        
    }
    Json::Reader reader;
    Json::Value root;
    if (!reader.parse(ifs, root)) {
        printf("Failed to parse activator_config.json\n");
        return -1;
    }
    ifs.close();

    const qfloat alpha = float_to_qfloat(root["extra_data"]["alpha_of_LeakyReLU"].asFloat());
    const qfloat allowedEpsion = float_to_qfloat(root["allowedEpsion"].asFloat());

    //测试

    p_matrix_qfloat input = alloc_matrix_qfloat();
    p_matrix_qfloat output = alloc_matrix_qfloat();

    uint32_t vectorsize = root["VectorSize"].asInt();

    matrix_qfloat_init(input, vectorsize, 1,0);
    matrix_qfloat_init(output, vectorsize, 1,0);
    
    printf("RawData\n");
    for(int i = 0; i < vectorsize; i++){
        input->data[i] = root["RawData"][i].asInt();
        printf("%.2f ", qfloat_to_float(input->data[i]));
    }
    printf("\n");

    int total_tests = 0;
    int failed_tests = 0;
    
    for(auto it : act_type_map){
        get_act_func(it.first, input, output, alpha);
        int error_count = 0;
        
        printf("\nTesting %s activation function:\n", it.second);
        printf("Input\t\tExpected\t\tActual\t\tError\n");
        
        for(int i = 0; i < vectorsize; i++){
            qfloat expected = root["excepteds"][it.second][i].asInt();
            qfloat actual = output->data[i];
            qfloat error = abs(expected - actual);
            
            printf("%8.4f\t%8.4f\t%8.4f\t%8.4f\n", 
                  qfloat_to_float(input->data[i]),
                  qfloat_to_float(expected),
                  qfloat_to_float(actual),
                  qfloat_to_float(error));
            
            if(error > allowedEpsion){
                error_count++;
            }
            total_tests++;
        }
        
        float error_rate = (float)error_count / vectorsize * 100;
        printf("\n%s Test Summary:\n", it.second);
        printf("Total samples: %d\n", vectorsize);
        printf("Failed samples: %d (%.2f%%)\n", error_count, error_rate);
        
        if(error_count > 0){
            failed_tests++;
            printf("TEST FAILED: Some samples exceed allowed error margin\n");
        } else {
            printf("TEST PASSED: All samples within allowed error margin\n");
        }
        printf("----------------------------------------\n");
    }
    
    printf("\nFinal Test Summary:\n");
    printf("Total activation functions tested: %lu\n", act_type_map.size());
    printf("Failed activation functions: %d\n", failed_tests);
    printf("Total samples tested: %d\n", total_tests);
    printf("Overall test %s\n", failed_tests == 0 ? "PASSED" : "FAILED");

    return 0;
}