//
//  arb.c
//  helloworld
//
//  Created by YangZiqing on 2017/7/11.
//  Copyright © 2017年 YangZiqing. All rights reserved.
//

#include "arb.h"
#include <stdio.h>

int move_to(int pos, int x, int y, int width, int height){
    int size = width * height;
    if ((pos/width)!=((pos+x)/width)) return -1;//水平方向越界
    else if ((pos/size)!=((pos+y*width)/size)) return -1; //垂直方向越界
    else
        return pos+(width*y)+x;
}

_Bool check_row(int * board, int pos, int width, int height){
    //row check
    int count = 1;
    int color = board[pos];
    int left=pos;
    int right=pos;

    if (board[pos]==0) return FALSE; //No stone here

    while(TRUE){
        left = move_to(left,-1,0,width,height);
        if ((left>=0) && board[left]==color) count+=1;
        else break;
    }
    while(TRUE){
        right = move_to(right,1,0,width,height);
        if ((right>=0) && board[right]==color) count+=1;
        else break;
    }
    if (count>=5) return TRUE;
    else return FALSE;
}

_Bool check_col(int * board, int pos, int width, int height){
    //column check
    int count = 1;
    int color = board[pos];
    int up = pos;
    int down = pos;
    
    if (board[pos]==0) return FALSE;

    while(TRUE){
        up = move_to(up,0,-1,width,height);
        if ((up>=0)&& board[up]==color) count+=1;
        else break;
    }
    while(TRUE){
        down = move_to(down,0,1,width,height);
        if ((down>=0) && board[down]==color) count+=1;
        else break;
    }
    if (count>=5) return TRUE;
    else return FALSE;
}

_Bool check_diag1(int * board, int pos, int width, int height){
    //diagnal direction check
    int count = 1;
    int color = board[pos];
    int ur = pos;
    int dl = pos;
    
    if (board[pos]==0) return FALSE;

    while(TRUE){
        ur = move_to(ur,1,-1,width,height);
        if ((ur>=0)&& board[ur]==color) count+=1;
        else break;
    }
    while(TRUE){
        dl = move_to(dl,-1,1,width,height);
        if ((dl>=0) && board[dl]==color) count+=1;
        else break;
    }
    if (count>=5) return TRUE;
    else return FALSE;
}

_Bool check_diag2(int * board, int pos, int width, int height){
    //diagnal direction check
    int count = 1;
    int color = board[pos];
    int ul=pos;
    int dr=pos;
    
    if (board[pos]==0) return FALSE;

    while(TRUE){
        ul = move_to(ul,-1,-1,width,height);
        if ((ul>=0)&& board[ul]==color) count+=1;
        else break;
    }
    while(TRUE){
        dr = move_to(dr,1,1,width,height);
        if ((dr>=0) && board[dr]==color) count+=1;
        else break;
    }
    if (count>=5) return TRUE;
    else return FALSE;
}

_Bool check(int *board, int pos, int width, int height){
    func_array checks = {check_row,check_col,check_diag1,check_diag2};
    for (int i=0;i<4;i++){
        if(checks[i](board,pos,width,height)) return TRUE;
    }
    return FALSE;
}
