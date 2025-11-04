#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;


void printEndTime (const string& str){
    int hour, minute;
    int hour_1=0, minute_1=0,a,b;
    string time,useless;
    istringstream input(str);
    input >> hour;
    input.ignore(1); // Ignore the colon
    input >> minute;
    input >> time; // Read the rest of the string
    cout << "Start time: " << hour << ":" << minute <<" "<< time << endl;
    while(1){
        input >> a;
        if(!input.fail()){
            b=a; 
        }
        if(input.fail()){
            input.clear(); // Clear the fail state
            input >> useless;
            cout << "Useless: " << useless << endl;
            if ((useless == "hour") || (useless == "hours")){
               hour_1 = b; 
            }       
            if (useless == "minutes"){
                minute_1 = b;
            }     
        }  
            
        if (input.eof()) {
            break;
        }
    } 
    // Adjust hour to be within 0-23
    cout << "End time: " << hour_1 << ":" << minute_1 << endl;
    if (minute_1+ minute >= 60) {
        hour_1 += (minute_1 + minute) / 60;
        minute = (minute_1 + minute) % 60;
    } else {
        minute += minute_1;
    }
    hour += hour_1;
    if (hour >= 24) {
        hour = hour % 24; // Wrap around if hour exceeds 23
    }
    cout << "Input time: " << hour << ":" << minute << endl;
}

 void badwelcomeprogram() {
    auto val=15;
    string name,response;
    int age;
    cout << "Welcome to the program!" << endl;
    cout << "Please enter your name: ";
    cin >> name;
    cout << "Hello, " << name << "!" << endl;
    cout << "How old are you? ";
    cin >> age;
    cout << "You are " << age << " years old." << endl;
    cout << "Do you like this program? (yes/no): ";
    if (cin.fail()) cin.clear(); // Clear the fail state if any
    cin >> response;
    cout << "You responded: " << response << endl;
    
    
}

void printVec(const vector<string>& vec) {
    cout << "Vector contents: ";
    for (auto str : vec) {
        cout << str << " ";
    }
    cout << endl;
}

int main() {
    // ostringstream output("Hello, World!",ostringstream::ate);
    // cout << output.str() << endl;

    // output<< 16.9 <<'a';
    // cout << output.str() << endl;

    // string str = "10:30 AM \n 1 hour 20 minutes";
    // cout << "Input string: " << str << endl;
    // printEndTime(str);
    // badwelcomeprogram();
    // auto prices= make_pair(10.5, 20.0);
    // cout << "Prices "<< prices.first << " and " << prices.second << endl;

    vector<string> vec;
    vec.push_back("Hello");
    vec.push_back("World");
    vec.push_back("!");

    printVec(vec);
    cout << vec[100]<<endl;
    cout << "Hello! I've reached this point" << endl;
    return 0;
}