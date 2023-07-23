#pragma once

/* 三元组 */
struct Triple {
    int i, j;   // 表示元素的行列值
    int e;      // 表示元素的值
};

/* 行逻辑链接顺序表 */
struct RLSMatrix {
    Triple data[1000];      // 非零元素的三元组表
    int rowOffset[1000];    // 各行第一个非零元素在data中的索引表
    int mu, nu, tu;         // m -> 行数, n -> 列数, t -> 非零元素的个数
};