#pragma once

#include "ldbc.hpp"
#include <algorithm>
#include <cmath>
#include <limits.h>

#define COMMON_PATH "/home/liupeng/data/LdbcDataset/" 

void saveWCCresult(string path, vector<vector<int>> & ivec);

bool compare(std::vector<int>& a, std::vector<int>& b) {
    return a[0] < b[0];
}

void bfs_ldbc_checker(LDBC<double>& graph, std::vector<int>& cpu_res, int & is_pass) {
    
    int size = cpu_res.size();

    if (size != graph.V) {
        std::cout << "Size of BFS results is not equal to the number of vertices!" << std::endl;
        return;
    }

    // std::string base_line_file = "../results/" + graph.vertex_file;
    std::string base_line_file = "/home/liupeng/data/LdbcDataset/" + graph.vertex_file;
    // remove the last two char

    base_line_file.pop_back();
    base_line_file.pop_back();

    base_line_file += "-BFS";

    std::ifstream base_line(base_line_file);

    if (!base_line.is_open()) {
        std::cout << "Baseline file not found!" << std::endl;
        return;
    }

    int id = 0;
    std::string line;
    while (std::getline(base_line, line)) {
        std::vector<std::string> tokens;
        tokens = parse_string(line, " ");
        if (tokens.size() != 2) {
            std::cout << "Baseline file format error!" << std::endl;
            base_line.close();
            return;
        }
        if (id >= size) {
            std::cout << "Size of baseline file is larger than the result!" << std::endl;
            base_line.close();
            return;
        }
        if (graph.vertex_str_to_id.find(tokens[0]) == graph.vertex_str_to_id.end()) {
            std::cout << "Baseline file contains a vertex that is not in the graph!" << std::endl;
            base_line.close();
            return;
        }
        int v_id = graph.vertex_str_to_id[tokens[0]];
        if (cpu_res[v_id] != std::stol(tokens[1])) {
            if(!(cpu_res[v_id] == INT_MAX && std::stol(tokens[1]) == LLONG_MAX)){
                std::cout << "Baseline file and GPU BFS results are not the same!" << std::endl;
                std::cout << "Baseline file: " << tokens[0] << " " << tokens[1] << std::endl;
                base_line.close();
                return;
            }
        }
        id++;
    }
    if (id != size) {
        std::cout << "Size of baseline file is smaller than the result!" << std::endl;
        base_line.close();
        return;
    }

    std::cout << "BFS results are correct!" << std::endl;
    is_pass = 1;
    base_line.close();
}

void wcc_ldbc_checker(LDBC<double>& graph, std::vector<std::vector<int>>& cpu_res, int & is_pass) {

    int size = cpu_res.size();

    for (auto &v : cpu_res) {
        if (!v.size()) {
            std::cout << "One of CPU WCC results is empty!" << std::endl;
            return;
        }
        std::sort(v.begin(), v.end());
    }

    
    std::sort(cpu_res.begin(), cpu_res.end(), compare);


    // std::string base_line_file = "../results/" + graph.vertex_file;
    std::string base_line_file = "/home/liupeng/data/LdbcDataset/" + graph.vertex_file;

    base_line_file.pop_back();
    base_line_file.pop_back();

    base_line_file += "-WCC";

    std::ifstream base_line(base_line_file);

    if (!base_line.is_open()) {
        std::cout << "Baseline file not found!" << std::endl;
        return;
    }

    std::vector<std::vector<int>> base_res;
    std::vector<int> component;

    component.resize(graph.V, 0);

    std::string line;

    while (std::getline(base_line, line)) {
        std::vector<std::string> tokens;
        tokens = parse_string(line, " ");
        if (tokens.size() != 2) {
            std::cout << "Baseline file format error!" << std::endl;
            base_line.close();
            return;
        }
        //store baseline file per row value to component
        component[graph.vertex_str_to_id[tokens[0]]] = graph.vertex_str_to_id[tokens[1]];
    }

    std::vector<std::vector<int>> componentLists(graph.V);

    for (int i = 0; i < graph.V; i++) {
        componentLists[component[i]].push_back(i);
    }

    for (int i = 0; i < graph.V; i++) {
		if (componentLists[i].size() > 0)
			base_res.push_back(componentLists[i]);
	}


    for (auto &v : base_res) {
        if (!v.size()) {
            std::cout << "One of baseline WCC results is empty!" << std::endl;
            base_line.close();
            return;
        }
        std::sort(v.begin(), v.end());
    }

    std::sort(base_res.begin(), base_res.end(), compare);

    for (int i = 0; i < size; i++) {
        if (base_res[i].size() != cpu_res[i].size()) {
            std::cout << "Baseline file and GPU WCC results are not the same!" << std::endl;
            std::cout << "Baseline file: " << graph.vertex_id_to_str[i] << " " << base_res[i][0] << std::endl;
            std::cout << "CPU WCC result: " << graph.vertex_id_to_str[cpu_res[i][0]] << " " << cpu_res[i][0] << std::endl;
            base_line.close();
            return;
        }
        for (int j = 0; j < base_res[i].size(); j++) {
            if (base_res[i][j] != cpu_res[i][j]) {
                std::cout << "Baseline file and GPU WCC results are not the same!" << std::endl;
                std::cout << "Difference at: " << graph.vertex_id_to_str[base_res[i][j]] << " " << graph.vertex_id_to_str[cpu_res[i][j]] << std::endl;
                base_line.close();
                return;
            }
        }
    }

    std::cout << "WCC results are correct!" << std::endl;
    is_pass = 1;
    base_line.close();
}

void sssp_ldbc_checker(LDBC<double>& graph, std::vector<double>& cpu_res, int & is_pass) {
    
    int size = cpu_res.size();

    if (size != graph.V) {
        std::cout << "Size of SSSP results is not equal to the number of vertices!" << std::endl;
        return;
    }

    // std::string base_line_file = "../results/" + graph.vertex_file;
    std::string base_line_file = "/home/liupeng/data/LdbcDataset/" + graph.vertex_file;

    // remove the last two char

    base_line_file.pop_back();
    base_line_file.pop_back();

    base_line_file += "-SSSP";

    std::ifstream base_line(base_line_file);

    if (!base_line.is_open()) {
        std::cout << "Baseline file not found!" << std::endl;
        return;
    }

    int id = 0;
    std::string line;
    while (std::getline(base_line, line)) {
        std::vector<std::string> tokens;
        tokens = parse_string(line, " ");
        if (tokens.size() != 2) {
            std::cout << "Baseline file format error!" << std::endl;
            base_line.close();
            return;
        }
        if (id >= size) {
            std::cout << "Size of baseline file is larger than the result!" << std::endl;
            base_line.close();
            return;
        }

        if (graph.vertex_str_to_id.find(tokens[0]) == graph.vertex_str_to_id.end()) {
            std::cout << "Baseline file contains a vertex that is not in the graph!" << std::endl;
            base_line.close();
            return;
        }
        int v_id = graph.vertex_str_to_id[tokens[0]];

        if (fabs(cpu_res[v_id] - std::stod(tokens[1])) > 1e-4) {
            std::cout << "Baseline file and GPU SSSP results are not the same!" << std::endl;
            std::cout << "Baseline file: " << tokens[0] << " " << tokens[1] << std::endl;
            std::cout << "GPU SSSP result: " << graph.vertex_id_to_str[v_id] << " " << cpu_res[v_id] << std::endl;
            base_line.close();
            return;
        }
        id++;
    }
    if (id != size) {
        std::cout << "Size of baseline file is smaller than the result!" << std::endl;
        base_line.close();
        return;
    }

    std::cout << "SSSP results are correct!" << std::endl;
    is_pass = 1;
    base_line.close();
}

void pr_ldbc_checker(LDBC<double>& graph, std::vector<double>& cpu_res, int & is_pass) {
    
    int size = cpu_res.size();

    if (size != graph.V) {
        std::cout << "Size of PageRank results is not equal to the number of vertices!" << std::endl;
        return;
    }

    // std::string base_line_file = "../results/" + graph.vertex_file;
    std::string base_line_file = "/home/liupeng/data/LdbcDataset/" + graph.vertex_file;

    // remove the last two char

    base_line_file.pop_back();
    base_line_file.pop_back();

    base_line_file += "-PR";

    std::ifstream base_line(base_line_file);

    if (!base_line.is_open()) {
        std::cout << "Baseline file not found!" << std::endl;
        return;
    }

    int id = 0;
    std::string line;
    while (std::getline(base_line, line)) {
        std::vector<std::string> tokens;
        tokens = parse_string(line, " ");
        if (tokens.size() != 2) {
            std::cout << "Baseline file format error!" << std::endl;
            base_line.close();
            return;
        }
        if (id >= size) {
            std::cout << "Size of baseline file is larger than the result!" << std::endl;
            base_line.close();
            return;
        }

        if (graph.vertex_str_to_id.find(tokens[0]) == graph.vertex_str_to_id.end()) {
            std::cout << "Baseline file contains a vertex that is not in the graph!" << std::endl;
            base_line.close();
            return;
        }
        int v_id = graph.vertex_str_to_id[tokens[0]];

        if (fabs(cpu_res[v_id] - std::stod(tokens[1])) > 1e-4) {
            std::cout << "Baseline file and GPU PageRank results are not the same!" << std::endl;
            std::cout << "Baseline file: " << tokens[0] << " " << tokens[1] << std::endl;
            std::cout << "GPU PageRank result: " << graph.vertex_id_to_str[v_id] << " " << cpu_res[v_id] << std::endl;
            base_line.close();
            return;
        }
        id++;
    }
    if (id != size) {
        std::cout << "Size of baseline file is smaller than the result!" << std::endl;
        base_line.close();
        return;
    }

    std::cout << "PageRank results are correct!" << std::endl;
    is_pass = 1;
    base_line.close();
}


void lcc_ldbc_checker(LDBC<double>& graph, std::vector<double>& cpu_res, int & is_pass) {
    
    int size = cpu_res.size();

    if (size != graph.V) {
        std::cout << "Size of LCC results is not equal to the number of vertices!" << std::endl;
        return;
    }

    // std::string base_line_file = "../results/" + graph.vertex_file;
    std::string base_line_file = "/home/liupeng/data/LdbcDataset/" + graph.vertex_file;

    // remove the last two char

    base_line_file.pop_back();
    base_line_file.pop_back();

    base_line_file += "-LCC";

    std::ifstream base_line(base_line_file);

    if (!base_line.is_open()) {
        std::cout << "Baseline file not found!" << std::endl;
        return;
    }

    int id = 0;
    std::string line;
    while (std::getline(base_line, line)) {
        std::vector<std::string> tokens;
        tokens = parse_string(line, " ");
        if (tokens.size() != 2) {
            std::cout << "Baseline file format error!" << std::endl;
            base_line.close();
            return;
        }
        if (id >= size) {
            std::cout << "Size of baseline file is larger than the result!" << std::endl;
            base_line.close();
            return;
        }

        if (graph.vertex_str_to_id.find(tokens[0]) == graph.vertex_str_to_id.end()) {
            std::cout << "Baseline file contains a vertex that is not in the graph!" << std::endl;
            base_line.close();
            return;
        }
        int v_id = graph.vertex_str_to_id[tokens[0]];

        if (fabs(cpu_res[v_id] - std::stod(tokens[1])) > 1e-4) {
            std::cout << "Baseline file and GPU LCC results are not the same!" << std::endl;
            std::cout << "Baseline file: " << tokens[0] << " " << tokens[1] << std::endl;
            std::cout << "GPU LCC result: " << graph.vertex_id_to_str[v_id] << " " << cpu_res[v_id] << std::endl;
            base_line.close();
            return;
        }
        id++;
    }
    if (id != size) {
        std::cout << "Size of baseline file is smaller than the result!" << std::endl;
        base_line.close();
        return;
    }

    std::cout << "LCC results are correct!" << std::endl;
    is_pass = 1;
    base_line.close();
}

void cdlp_ldbc_check(LDBC<double>& graph, std::vector<string>& cpu_res, int & is_pass) {
    
    int size = cpu_res.size();

    if (size != graph.V) {
        std::cout << "Size of CDLP results is not equal to the number of vertices!" << std::endl;
        return;
    }


    // std::string base_line_file = "../results/" + graph.vertex_file;
    std::string base_line_file = "/home/liupeng/data/LdbcDataset/" + graph.vertex_file;

    // remove the last two char

    base_line_file.pop_back();
    base_line_file.pop_back();

    base_line_file += "-CDLP";

    std::ifstream base_line(base_line_file);

    if (!base_line.is_open()) {
        std::cout << "Baseline file not found!" << std::endl;
        return;
    }

    int id = 0;
    std::string line;
    while (std::getline(base_line, line)) {
        std::vector<std::string> tokens;
        tokens = parse_string(line, " ");
        if (tokens.size() != 2) {
            std::cout << "Baseline file format error!" << std::endl;
            base_line.close();
            return;
        }
        if (id >= size) {
            std::cout << "Size of baseline file is larger than the result!" << std::endl;
            base_line.close();
            return;
        }

        if (graph.vertex_str_to_id.find(tokens[0]) == graph.vertex_str_to_id.end()) {
            std::cout << "Baseline file contains a vertex that is not in the graph!" << std::endl;
            base_line.close();
            return;
        }
        int v_id = graph.vertex_str_to_id[tokens[0]];
        
        if (cpu_res[v_id] != tokens[1]) {
            std::cout << "Baseline file and GPU CDLP results are not the same!" << std::endl;
            std::cout << "Baseline file: " << tokens[0] << " " << tokens[1] << std::endl;
            std::cout << "CPU CDLP result: " << cpu_res[v_id] << std::endl;
            base_line.close();
            return;
        }
        id++;
    }
    if (id != size) {
        std::cout << "Size of baseline file is smaller than the result!" << std::endl;
        base_line.close();
        return;
    }

    std::cout << "CDLP results are correct!" << std::endl;
    is_pass = 1;
    base_line.close();
}

void saveWCCresult(string path, vector<vector<int>> & ivec){
    // 打开文件
    std::ofstream outFile(path);

    if (!outFile) {
        std::cerr << "Error: Unable to open the file." << std::endl;
        // return 1;
    }

    for (const auto& row : ivec) {
        for (const auto& value : row) {
            outFile << value <<" ";
        }
        outFile << std::endl;
    }

    outFile.close();

    std::cout << "Data has been saved to data.txt" << std::endl;
}