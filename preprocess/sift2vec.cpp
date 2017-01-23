#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <string.h>
using namespace std;
#define MAX_D 25
struct vc
{
	float dis[MAX_D];
	vc()
	{
		memset(dis,0,sizeof(dis));
	}
}tv;
map<string,vc>maplive;
int main()
{
	ifstream fin;
	fin.open("glove.twitter.27B.25d.txt",ios::in);
	string topic;
	while(fin>>topic)
	{
		for(int i =0;i<MAX_D;i++)
			fin>>tv.dis[i];
		maplive.insert(pair<string,vc>(topic,tv));
	}
	fin.close();
	cout<<"end read"<<endl;
	map<string,vc>::iterator l_it;
	fin.open("all_small.txt",ios::in);
	ofstream fout;
	fout.open("vec_res_2.txt",ios::out);
	int longest = 0;
	int char_num=0;
	int len_num;
	int len_id=0;
	int have_ans=0;
	int miss_num = 0;
	while(fin>>topic)
	{
		//cout<<topic<<endl;
		//getchar();
		//system("pause");
		//cout<<topic<<' ';
		if(topic == "hrbans")
		{
			fout<<"ans"<<endl;
			//have_ans++;
			continue;
		}
		else if(topic == "lable" )
		{
			//len_id++;
			//fout<<"lable"<<' ';
			fin>>topic;
			fout<<topic<<endl;
			if(char_num>longest)
			{
				longest=char_num;
				//len_num=len_id;
			}
			char_num=0;
		}
		else
		{
			
			l_it=maplive.find(topic);//返回的是一个指针
			if(l_it==maplive.end())
			{
				miss_num++;
				continue;
			}
			else
			{
				char_num++;
				for(int i = 0;i<MAX_D;i++)
					fout<<l_it->second.dis[i]<<' ';
				fout<<endl;
			}
		}
	}
	fout.close();
	fin.close();
	cout<<"longest:"<<longest<<endl;
	cout<<"miss num"<<miss_num<<endl;
	//cout<<"len_num"<<len_num<<endl;
	//cout<<"have_ans"<<have_ans<<endl;
	return 0;
}