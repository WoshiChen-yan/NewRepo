class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int num=nums.size();
        int val=0;
        int temp;
        int i=0;
        int j=i;
        while(j<num){
                val=val+nums[j];
                if(val>=target){
                    break;
                }
                j++;     
        }
        temp=j-i+1;
        while(i!=j){
            if(val<target){
                if(j<num-1){
                  j++;
                  val=val+nums[j]; 
                }
                else{
                    if(temp==num){
                        temp=0;
                    } 
                    break;
                } 
            }
            else{
                if(temp>(j-i+1)){
                    temp=j-i+1;
                }
                val=val-nums[i];
                i++;
            }
        }
        return temp;
    }
};