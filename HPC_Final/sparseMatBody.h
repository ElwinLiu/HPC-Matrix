#pragma once

/* ��Ԫ�� */
struct Triple {
    int i, j;   // ��ʾԪ�ص�����ֵ
    int e;      // ��ʾԪ�ص�ֵ
};

/* ���߼�����˳��� */
struct RLSMatrix {
    Triple data[1000];      // ����Ԫ�ص���Ԫ���
    int rowOffset[1000];    // ���е�һ������Ԫ����data�е�������
    int mu, nu, tu;         // m -> ����, n -> ����, t -> ����Ԫ�صĸ���
};