#include "mlp.h"
#include "../threadpool/threadpool.h"

/*DOC
多线程优化指导
对于mlp运算的链式特性,多线程优化将采用规约方式计算并用线程池整合

如:

        W        *      x   +   B
| 1, 2, 3, 4, 5, 6 |  | 1 |   | 3 |
| 1, 2, 3, 4, 5, 6 |  | 2 |   | 6 |
| 1, 2, 3, 4, 5, 6 |  | 3 |   | 9 |
| 1, 2, 3, 4, 5, 6 |  | 4 |   | 2 |
| 1, 2, 3, 4, 5, 6 |  | 5 |   | 4 |
| 1, 2, 3, 4, 5, 6 |  | 6 |   | 6 |

<mlpxxx_exec_mt>对于一般前向,其基本流程可分解为
sub1: sum_part_wx(w,x,r,c, delta_c) = sum(i = c, delta_c + c - 1, w[r, i] * x[i])
sub2: resu = call sub1 * ceil(w_c / delta_c) ; resu += B[i]; ac(resu)
sub3: sub2(i) * out_dim

<mlptrainer_backward_mt>对于一般反向,权重,偏置更新不影响反传,可以:
grad -> grad1 -> ... -> grad(n-1) -> gradn
     |        |                   |
     v        v                   v
     w,b      w,b                 w,b


<mlptrainer_collector_xxx_mt>采用一般多线程模式即可

验证: 一般计算,指定 W,x,b 值并让其计算, 设置 actp = pass, 检测多线程结果是否符合预期；
    collector的设置两组,第一组trainer_backward(lr任意),第二组trainer_backward(lr = 0)&savegrads&total_backward(lr与第一组一致),比较两组再次trainer_execute输出结果,两组输出相同则通过
*/
