#include <iostream>
#include <fstream>
using namespace std;
int main()
{
	ifstream fin;
	fin.open("Que_Ans.txt",ios::in);
	//fin.open("test.txt",ios::in);
	char tmp;
	ofstream fout;
	fout.open("all_small.txt",ios::out);
	while(fin.get(tmp))
	{
		if(tmp>='A'&&tmp<='Z')
			//fout<<(char)('a'+1);
			fout<<(char)('a'+(tmp-(char)'A'));
		else if(tmp>='a'&&tmp<='z')
			fout<<tmp;
		else if(tmp==' '||tmp=='\n'||tmp==' ')
			fout<<tmp;
		else
			fout<<' ';
	}
	fin.close();
	fout.close();
	return 0;
}