1. cuptiActivityEnable
   * cupti activity 指定类型后直接开启activity统计，无需其他start接口。
   * 本地记录用户指定启用的activity kind信息，做合法性检查
   * 不调用cuptiActivityRegisterCallbacks注册回调函数，也可以调用cuptiActivityEnable，
   * runtime/driver采集到数据后通过rdma向host buffer刷新，如果host buffer不存在则调用用户提供的callback函数分配host buffer。
   * 启动独立线程，监控已分配host buffer的状态，当buffer满时调用用户提供的callback函数处理buffer数据，之后放弃当前buffer的控制权，用户负责释放buffer
   * 每次调用用户接口处理buffer前先申请一块新的buffer供使用。
   * buffer满或者达到用户设置的最大数量时，调用用户注册的buffer complete回调函数。
   * buffer request在主线程中调用，buffer complete在新建的线程中调用。
   * buffer中可以保存多个类型的activity数据
   * 使能cuptiActivityEnable时不会创建context
   * 调用cuptiActivityEnable后不会马上申请host buffer，而是在收集到数据后再分配。

2. cuptiSubscribe
   * 同一时刻只能有一个订阅者
   * 支持回调函数为NULL

3. Correlation ID
   * 所有runtime操作会内部分配不重复的Correlation id，launch操作和对应的kernel操作共用同一个id
   * 使用cuptiActivityPushExternalCorrelationId和cuptiActivityPopExternalCorrelationId，并且使能CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION时，cupti会创建stack保存External Id，每一个runtime&driver操作调用时创建Correlation Record，记录内部Correlation id和外部External Id的对应关系。
   * External Id使用栈顶的那个。
   * cuptiActivityPushExternalCorrelationId区分不同类型，每一种类型用独立stack记录。如果同时向3个类型执行cuptiActivityPushExternalCorrelationId操作，则每个runtime操作会创建3个 Correlation Record，对应3个类型的stack。

