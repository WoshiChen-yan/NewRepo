class Solution {
public:
    vector<int> sortedSquares(vector<int>& nums) {
            int i=0;
            for(i=-0; i<nums.size();i++){
                if (nums[i]>=0) break;
            }
            vector<int> val;
            int k;
            int j;
            j=i;
            i--;
            while((i>=0)&&(j<nums.size())){
               
                if(((nums[j]*nums[j])>=(nums[i]*nums[i]))){
                    
                    val.push_back(nums[i]*nums[i]);
                    i--;
                }
                else{
                    val.push_back(nums[j]*nums[j]);
                    j++;
                }  
            }
        while (i >= 0) {
            val.push_back(nums[i] * nums[i]);
            i--;
        }
        while (j < nums.size()) {
            val.push_back(nums[j] * nums[j]);
            j++;
        }       
        return val;     
       
    }
};