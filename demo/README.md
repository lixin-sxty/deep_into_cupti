1. cuptiActivityEnable
   * cupti activity 指定类型后直接开启trace统计，没有其他start接口。
   * 不调用cuptiActivityRegisterCallbacks注册回调函数，也可以调用cuptiActivityEnable，不确定是否实际进行了数据收集。
   * 调用cuptiActivityEnable后不会马上申请host buffer，而是在收集到数据后再分配。
   * buffer request在主线程中调用，buffer complete在其他线程中调用。
   * 使能cuptiActivityEnable时不会创建context，不占用显存

2. cuptiSubscribe
   * 同一时刻只能有一个订阅者
   * 支持回调函数为NULL
