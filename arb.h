//
//  arb.h
//  helloworld
//
//  Created by YangZiqing on 2017/7/11.
//  Copyright © 2017年 YangZiqing. All rights reserved.
//

#ifndef arb_h
#define arb_h

#define TRUE 1
#define FALSE 0


#include <stdio.h>
typedef _Bool (*func_array[])(int *,int,int,int);
_Bool check(int *,int pos, int width, int height);

#endif /* arb_h */
