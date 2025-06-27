#include <cupti.h>
#include <stdio.h>
#include <utils/utils.h>

void CUPTIAPI callback(void *userdata, CUpti_CallbackDomain domain,
                       CUpti_CallbackId cbid,
                       const CUpti_CallbackData *cbInfo) {
  SPDLOG_INFO("[Callback1] Domain {}, Callback ID {}", (int)domain, (int)cbid);
}

int main() {
  CUpti_SubscriberHandle subscriber1 = NULL, subscriber2;
  CUptiResult res;

  // 第一次订阅
  CUPTI_CALL(cuptiSubscribe(&subscriber1, NULL, NULL));
  SPDLOG_INFO("Subscribe 1: {}", (void *)subscriber1);

  // 只支持一个订阅者
  auto state = cuptiSubscribe(&subscriber2, (CUpti_CallbackFunc)callback, NULL);
  if (state == CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED) {
    SPDLOG_WARN("Multiple subscribers not supported");
  }

  // 清理第一个订阅者
  CUPTI_CALL(cuptiUnsubscribe(subscriber1));
  SPDLOG_INFO("Unsubscribe 1: {}", (void *)subscriber1);

  return 0;
}
