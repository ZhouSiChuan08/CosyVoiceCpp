#ifndef THREADPOOL_H
#define THREADPOOL_H
#include <iostream>
#include <thread>
#include <mutex>
#include <string>
#include <fstream>
#include <condition_variable>
#include <queue>
#include <deque>
#include <functional>  //传函数
class ThreadPool
{
public:
    ThreadPool(int num_threads) : stop(false)
    {
        for (int i = 0; i < num_threads; i++)
        {
            threads.emplace_back([this](){  //线程不支持拷贝移动，所以使用emplace_back，无需创建临时对象
                while (true)
                {
                    std::unique_lock<std::mutex> lock(mtx);  //加锁
                    condition.wait(lock, [this](){
                        return !tasks.empty() || stop;  //等待条件变量，直到有任务或线程池停止
                    });
                    if (stop && tasks.empty())  //线程池停止 且任务队列为空
                    {
                        return;
                    }
                    //不阻塞 取出任务并执行
                    std::function<void()> task(std::move(tasks.front()));  //移动任务，减少拷贝开销
                    tasks.pop();
                    lock.unlock();  //解锁
                    task();  //执行任务  
                }
                
            });
        } 
    }
    ~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lock(mtx);  //加锁
            stop = true;  //设置线程池停止标志
        }
        condition.notify_all();  //通知所有线程， 完成剩余任务
        for (auto& t: threads)  //对threads 使用引用遍历，减少拷贝开销
        {
            t.join();  //等待所有线程完成
        }
        std::cout << "线程池已关闭" << std::endl;
    }
    template<class F, class... Args>
    void enqueue(F &&f, Args &&... args)  //&&表示通用引用，可左可右
    {
        std::function<void()> task = 
            std::bind(std::forward<F>(f), std::forward<Args>(args)...);  //绑定函数
            {
                std::unique_lock<std::mutex> lock(mtx);  //加锁
                tasks.emplace(std::move(task));  //添加任务
            }
            //任务已添加，通知消费者完成任务
            condition.notify_one();  //通知一个线程，完成任务
    }
        
private:
    std::vector<std::thread> threads;  //线程容器
    std::queue<std::function<void()>> tasks;  //任务队列
    std::mutex mtx;  //互斥锁
    std::condition_variable condition;  //条件变量
    bool stop;  //线程池停止标志
};

#endif // THREADPOOL_H