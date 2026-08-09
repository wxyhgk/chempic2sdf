# G3 P6-E9 Structural Residual Arms

> 日期：2026-08-09
>
> 状态：已预注册，等待训练

固定P6-E1标准化peak prior与cap 2.0，三臂均从零residual严格复现prior开始：global、local17、
multiscale(17+41)。learned branch不读取peak logit；最终分数为`prior+2*tanh(residual)`。

train使用5,171 hard及按variant/role/degree匹配的等量easy，每hard 192次曝光、32:32 batch、D4、
AdamW `3e-4`。hard优化6px等价集合NLL；easy优化prior KL、0.25 winner hinge及0.01 residual L2。
三臂只读取calibration选模：先要求全部easy harmful=0，再按hard恢复率、MAE、参数量排序。两个
holdout cache保持000，dev、confirm和外部集不读取。每个实际模型均保存checkpoint并打同名tag。
