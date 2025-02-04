Building server for Tracy v0.11.1 hits build errors  
for the library oneTBB (tbb@2021.12.0 (v2021.12.0-rc2)):

----
In file included from /usr/include/c++/13/atomic:41,
                 from /home/nfginola/ws/tracy/profiler/build/_deps/tbb-src/src/tbb/../../include/oneapi/tbb/detail/_utils.h:22,
                 from /home/nfginola/ws/tracy/profiler/build/_deps/tbb-src/src/tbb/task_dispatcher.h:20,
                 from /home/nfginola/ws/tracy/profiler/build/_deps/tbb-src/src/tbb/arena.cpp:17:
In member function ‘void std::__atomic_base<_IntTp>::store(__int_type, std::memory_order) [with _ITp = bool]’,                  
    inlined from ‘void std::atomic<bool>::store(bool, std::memory_order)’ at /usr/include/c++/13/atomic:104:20,               
    inlined from ‘void tbb::detail::r1::concurrent_monitor_base<Context>::abort_all_relaxed() [with Context = long unsigned int]’ at /home/nfginola/ws/tracy/profiler/build/_deps/tbb-src/src/tbb/c
oncurrent_monitor.h:440:53,                                                                                                                                                                        
    inlined from ‘void tbb::detail::r1::concurrent_monitor_base<Context>::abort_all() [with Context = long unsigned int]’ at /home/nfginola/ws/tracy/profiler/build/_deps/tbb-src/src/tbb/concurren
t_monitor.h:423:26,                                                                                                                                                                                
    inlined from ‘void tbb::detail::r1::concurrent_monitor_base<Context>::destroy() [with Context = long unsigned int]’ at /home/nfginola/ws/tracy/profiler/build/_deps/tbb-src/src/tbb/concurrent_
monitor.h:456:24,                                                                                
    inlined from ‘tbb::detail::r1::concurrent_monitor::~concurrent_monitor()’ at /home/nfginola/ws/tracy/profiler/build/_deps/tbb-src/src/tbb/concurrent_monitor.h:487:16,
    inlined from ‘tbb::detail::r1::arena_base::~arena_base()’ at /home/nfginola/ws/tracy/profiler/build/_deps/tbb-src/src/tbb/arena.h:186:8:
/usr/include/c++/13/bits/atomic_base.h:481:25: error: ‘void __atomic_store_1(volatile void*, unsigned char, int)’ writing 1 byte into a region of size 0 overflows the destination [-Werror=stringo
p-overflow=]   
  481 |         __atomic_store_n(&_M_i, __i, int(__m));                                                                                                                                            
      |         ~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~                                           
compilation terminated due to -Wfatal-errors
----

This doesn't seem to be present when building server on current Tracy master (v0.12.1).
For now, workaround is to build with disabled '-Wfatal-errors' which can be done through "-DTBB_STRICT=OFF"

For building server (clone tracy and checkout v0.11.1):

// LEGACY on if using X11 (check tracy manual pdf)
// TBB_STRICT off (https://github.com/uxlfoundation/oneTBB/issues/1127#issuecomment-1940547584)
cmake -B profiler/build -S profiler -DCMAKE_BUILD_TYPE=Release -DLEGACY=ON -DTBB_STRICT=OFF
cmake --build profiler/build --config Release --parallel









