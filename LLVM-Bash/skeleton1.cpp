#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/IR/InlineAsm.h"
#include <vector>
#include<fstream>
using namespace std;
using namespace llvm;

//
static LLVMContext Context;
struct SkeletonPass : public FunctionPass {
    static char ID;
    SkeletonPass() : FunctionPass(ID) {}
    map<string, BasicBlock*> map_n_b;
    vector< BasicBlock * > bb_vector;
    vector< BasicBlock * > bb_fakeins;//
    map<BasicBlock*, int> visit;
    map<string, string> map_b;
    map<string, string> map_b_r;//
    map<string, int> suc_num;//
    map<string, vector<string> > suc_str; //
    vector<pair<BasicBlock*, BasicBlock*>> cut_pair;//
    ReturnInst* returninst;//

    //
    string funcBname = "main";

    virtual bool runOnFunction(Function &F) {
//
      //
      //
      std::ifstream in_f;
      in_f.open(" ./passsource_target.txt");
      string match_l;
      match_l.clear();
      getline(in_f, match_l);
      string func_name;
      int pos=match_l.find(" ");
      func_name=match_l.substr(0, pos);
      funcBname=match_l.substr(pos+1);
      int pos_tmp = 0;
      pos_tmp=func_name.find("@");
      if(F.getName().data()!=func_name.substr(pos_tmp+1))
      {
        errs()<<"fail to transfer!"<<"\n";
        return false;
      }
      errs() << "src:" << F.getName() << " = " << func_name << "\n";
      cfg_transfer(F);
      //
      
      errs()<<"transfer finish"<<"\n";
      return false;
    }

    void select_ins(Function &F);
    void create_line(BasicBlock* cur_bb, string target_b, int num_s, Function &F);//
    void cfg_transfer(Function &F);
    bool dfs(BasicBlock* start, BasicBlock* end);
    void insert_fakeins(void);
};


void SkeletonPass::select_ins(Function &F)
{
  
  Function::iterator fi;
  Function::iterator bs, be;
  IRBuilder<> B(Context);
  int id=0;
  int count=0;
  int break_n=0;
  BasicBlock *cur_bb=NULL;

  vector< BasicBlock * > bb_vector;
  bb_vector.clear();
  for (bs=F.begin(), be=F.end(); bs!=be; bs++)
  {
    bb_vector.push_back(&*bs);
  }

  
  for (int i=0; i<bb_vector.size(); i++)
  {
      //
      //
      cur_bb=bb_vector[i];
      if(!cur_bb->getTerminator())
      {
        continue;
      }
      if(isa<ReturnInst>(cur_bb->getTerminator()))
      {
        errs()<<"return ins:"<<F.getName()<<" "<<bb_vector.size()<<"\n";
        continue;
      }
      if (isa<BranchInst>(cur_bb->getTerminator()))
      {
        BranchInst*br=cast<BranchInst>(cur_bb->getTerminator());
        if (br->isUnconditional())//
        {
          if(false)//
          {
            id=0;
            BasicBlock *new_bb=BasicBlock::Create(Context, "jmp", &F);
            BasicBlock *suc_bb=br->getSuccessor(id);
            br->setSuccessor(id, new_bb);
            B.SetInsertPoint(new_bb);
            B.CreateBr(suc_bb);
            if(PHINode* phi_ins=dyn_cast <PHINode>(suc_bb->begin()))//
            {
              phi_ins->replaceIncomingBlockWith(cur_bb, new_bb);
            }
          }
          else//
          {
            BasicBlock *new_bb=BasicBlock::Create(Context, "jmp", &F);
            BasicBlock *suc_bb=br->getSuccessor(0);
            cur_bb->getTerminator()->eraseFromParent();
            B.SetInsertPoint(cur_bb);
            Value* fake=ConstantInt::getFalse(Context);
            B.CreateCondBr(fake, new_bb, suc_bb);
            B.SetInsertPoint(new_bb);
            B.CreateBr(suc_bb);
            if(PHINode* phi_ins=dyn_cast <PHINode>(suc_bb->begin()))//
            {
              //
              Value* v_tmp=phi_ins->getIncomingValue(0);
              phi_ins->addIncoming(v_tmp, new_bb);
            }
          }
          //
        }
        else if (br->isConditional())//
        {
          if(true)//
          {
            id=1;
            BasicBlock *new_bb=BasicBlock::Create(Context, "jmp", &F);
            BasicBlock *suc_bb=br->getSuccessor(id);
            br->setSuccessor(id, new_bb);
            B.SetInsertPoint(new_bb);
            B.CreateBr(suc_bb);
            if(PHINode* phi_ins=dyn_cast <PHINode>(suc_bb->begin()))//
            {
              phi_ins->replaceIncomingBlockWith(cur_bb, new_bb);
            }
          }
          //
        }




      }
      //
      //
      //
      //
      //
      }
}
void SkeletonPass::create_line(BasicBlock* cur_bb, string target_b, int num_s, Function &F)//
{
  pair<BasicBlock*, BasicBlock*> p1;
  
  

  //

  Function::iterator fi;
  Function::iterator bs, be;
  IRBuilder<> B(Context);
  
  //
  if(!cur_bb->getTerminator())//
  {
    if(num_s==1)//
    {
      for(int i=0; i<num_s; i++)
      {
        //
        //
        string suc_name=suc_str[target_b][i];//
        string cor_name=map_b_r[suc_name];//
        BasicBlock* suc_bb=map_n_b[cor_name];
        
        B.SetInsertPoint(cur_bb);
        B.CreateBr(suc_bb);
        if(PHINode* phi_ins=dyn_cast <PHINode>(suc_bb->begin()))//
        {
          Value* v_tmp=phi_ins->getIncomingValue(0);
          phi_ins->addIncoming(v_tmp, cur_bb);
        }
      }
    }
    else if (num_s==2)//
    {
      B.SetInsertPoint(cur_bb);
      Value* fake=ConstantInt::getFalse(Context);
      B.CreateCondBr(fake, cur_bb, cur_bb);
      BranchInst*br = cast<BranchInst>(cur_bb->getTerminator());
      for(int i=0; i<num_s; i++)
      {
        //
        //
        string suc_name=suc_str[target_b][i];//
        string cor_name=map_b_r[suc_name];//
        BasicBlock* suc_bb=map_n_b[cor_name];
        br->setSuccessor(i, suc_bb);
        if(PHINode* phi_ins=dyn_cast <PHINode>(suc_bb->begin()))//
        {
          Value* v_tmp=phi_ins->getIncomingValue(0);
          phi_ins->addIncoming(v_tmp, cur_bb);
        }
      }
    }
    else if(num_s==0)//
    {
      errs()<<"encounter retrun 1\n";
      B.SetInsertPoint(cur_bb);
      //
      //
      //

      auto *new_inst = returninst->clone();
      cur_bb->getInstList().push_back(new_inst); 
      //
      //
      //

      //
      //
      //
    }
    else//
    {
      return;
    }
  }
  else if (isa<BranchInst>(cur_bb->getTerminator()))
  {
    BranchInst*br=cast<BranchInst>(cur_bb->getTerminator());
    if (br->isUnconditional())//
    {
      if(num_s==1)//
      {
        for(int i=0; i<num_s; i++)
        {
          //
          //
          string suc_name=suc_str[target_b][i];//
          string cor_name=map_b_r[suc_name];//
          BasicBlock* suc_bb=map_n_b[cor_name];
          
          BasicBlock *ori_suc_bb=br->getSuccessor(i);//
          if(ori_suc_bb==suc_bb)
          {
            break;
          }
          else
          {
            br->setSuccessor(i, suc_bb);
            //
            p1.first=cur_bb;
            p1.second=ori_suc_bb;
            cut_pair.push_back(p1);
            PHINode* phi_ins=dyn_cast <PHINode>(suc_bb->begin());
            
            if(suc_bb->size())
            {
              PHINode* phi_ins=dyn_cast <PHINode>(suc_bb->begin());
              if(phi_ins)
              {
                Value* v_tmp=phi_ins->getIncomingValue(0);
                phi_ins->addIncoming(v_tmp, cur_bb);
              }
            }
          }
          
        }
      }
      else if (num_s==2)//
      {
        //
        BasicBlock *ori_suc_bb=br->getSuccessor(0);
        cur_bb->getTerminator()->eraseFromParent();
        B.SetInsertPoint(cur_bb);
        Value* True=ConstantInt::getTrue(Context);
        Value* fake=ConstantInt::getFalse(Context);
        //
        //
        //
        int flag=0;
        for(int i=0; i<num_s; i++)
        {
          //
          //
          string suc_name=suc_str[target_b][i];//
          string cor_name=map_b_r[suc_name];//
          BasicBlock* suc_bb=map_n_b[cor_name];
          
          if(ori_suc_bb==suc_bb && i==0)
          {
            flag=1;
            
            //
            string suc_name=suc_str[target_b][1];
            string cor_name=map_b_r[suc_name];
            BasicBlock* suc_bb_next=map_n_b[cor_name];

            B.CreateCondBr(True, suc_bb, suc_bb_next);
            if(PHINode* phi_ins=dyn_cast <PHINode>(suc_bb->begin()))//
            {
              Value* v_tmp=phi_ins->getIncomingValue(0);
              phi_ins->addIncoming(v_tmp, cur_bb);
            }
            if(PHINode* phi_ins=dyn_cast <PHINode>(suc_bb_next->begin()))//
            {
              Value* v_tmp=phi_ins->getIncomingValue(0);
              phi_ins->addIncoming(v_tmp, cur_bb);
            }
            break;
          }
          else if(ori_suc_bb==suc_bb && i==1)
          {
            flag=1;
            //
            string suc_name=suc_str[target_b][0];
            string cor_name=map_b_r[suc_name];
            BasicBlock* suc_bb_front=map_n_b[cor_name];

            B.CreateCondBr(fake, suc_bb_front, suc_bb);
            if(PHINode* phi_ins=dyn_cast <PHINode>(suc_bb->begin()))//
            {
              Value* v_tmp=phi_ins->getIncomingValue(0);
              phi_ins->addIncoming(v_tmp, cur_bb);
            }
            if(PHINode* phi_ins=dyn_cast <PHINode>(suc_bb_front->begin()))//
            {
              Value* v_tmp=phi_ins->getIncomingValue(0);
              phi_ins->addIncoming(v_tmp, cur_bb);
            }
            break;
          }
        }
        if(flag==0)
        {
          string suc_name=suc_str[target_b][0];
          string cor_name=map_b_r[suc_name];
          BasicBlock* suc_bb_front=map_n_b[cor_name];

          suc_name=suc_str[target_b][1];
          cor_name=map_b_r[suc_name];
          BasicBlock* suc_bb=map_n_b[cor_name];
          B.CreateCondBr(fake, suc_bb_front, suc_bb);
          //
          p1.first=cur_bb;
          p1.second=ori_suc_bb;
          cut_pair.push_back(p1);

          if(PHINode* phi_ins=dyn_cast <PHINode>(suc_bb->begin()))//
          {
            Value* v_tmp=phi_ins->getIncomingValue(0);
            phi_ins->addIncoming(v_tmp, cur_bb);
          }
          if(PHINode* phi_ins=dyn_cast <PHINode>(suc_bb_front->begin()))//
          {
            Value* v_tmp=phi_ins->getIncomingValue(0);
            phi_ins->addIncoming(v_tmp, cur_bb);
          }
        }
      }
      else//
      {
        return;
      }
      //
    }
    else if (br->isConditional())//
    {
      if(num_s==2){
        BasicBlock *ori_suc_bb=br->getSuccessor(0);
        
        string suc_name=suc_str[target_b][1];//
        string cor_name=map_b_r[suc_name];//
        BasicBlock* suc_bb_next=map_n_b[cor_name];
        

        if(ori_suc_bb==suc_bb_next)
        {
          //
          suc_str[target_b][1]=suc_str[target_b][0];
          suc_str[target_b][0]=suc_name;
        }
        for(int i=0; i<num_s; i++)
        {
          //
          //
          string suc_name=suc_str[target_b][i];//
          string cor_name=map_b_r[suc_name];//
          BasicBlock* suc_bb=map_n_b[cor_name];
          
          BasicBlock *ori_suc_bb=br->getSuccessor(i);//
          if(ori_suc_bb==suc_bb)//
          {
            continue;
          }
          else
          {

            br->setSuccessor(i, suc_bb);
            //
            p1.first=cur_bb;
            p1.second=ori_suc_bb;
            cut_pair.push_back(p1);

            if(PHINode* phi_ins=dyn_cast <PHINode>(suc_bb->begin()))//
            {
              Value* v_tmp=phi_ins->getIncomingValue(0);
              phi_ins->addIncoming(v_tmp, cur_bb);
            }
          }
        }
      }
      else if(num_s>2){//
        return ;
      }
      //
    }
  }
  else if (isa<SwitchInst>(cur_bb->getTerminator()))
  {
    //
    return;
  }
  else if (isa<ReturnInst>(cur_bb->getTerminator()))
  {
    returninst=dyn_cast<ReturnInst>(cur_bb->getTerminator());
    if(num_s==1)//
    {
      for(int i=0; i<num_s; i++)
      {
        //
        //
        string suc_name=suc_str[target_b][i];//
        string cor_name=map_b_r[suc_name];//
        BasicBlock* suc_bb=map_n_b[cor_name];
        
        cur_bb->getTerminator()->eraseFromParent();
        B.SetInsertPoint(cur_bb);
        B.CreateBr(suc_bb);
        if(PHINode* phi_ins=dyn_cast <PHINode>(suc_bb->begin()))//
        {
          Value* v_tmp=phi_ins->getIncomingValue(0);
          phi_ins->addIncoming(v_tmp, cur_bb);
        }
      }
    }
    else if (num_s==2)//
    {
      //
      //
      cur_bb->getTerminator()->eraseFromParent();
      B.SetInsertPoint(cur_bb);
      Value* True=ConstantInt::getTrue(Context);
      Value* fake=ConstantInt::getFalse(Context);
      //
      //
      //
      
      string suc_name=suc_str[target_b][0];
      string cor_name=map_b_r[suc_name];
      BasicBlock* suc_bb_front=map_n_b[cor_name];

      suc_name=suc_str[target_b][1];
      cor_name=map_b_r[suc_name];
      BasicBlock* suc_bb=map_n_b[cor_name];
      B.CreateCondBr(fake, suc_bb_front, suc_bb);

      if(PHINode* phi_ins=dyn_cast <PHINode>(suc_bb->begin()))//
      {
        Value* v_tmp=phi_ins->getIncomingValue(0);
        phi_ins->addIncoming(v_tmp, cur_bb);
      }
      if(PHINode* phi_ins=dyn_cast <PHINode>(suc_bb_front->begin()))//
      {
        Value* v_tmp=phi_ins->getIncomingValue(0);
        phi_ins->addIncoming(v_tmp, cur_bb);
      }
    }
    else if(num_s==0)//
    {
      //
      //
      //
      //
      //
      //
      //
      
      //
      //
      //
      //
      //
      //

    }
    else//
    {
      return;
    }

  }
  return ;
}
      
void SkeletonPass::cfg_transfer(Function &F)
{
  //
  Function::iterator fi;
  Function::iterator bs, be;
  IRBuilder<> B(Context);
  int id=0;
  int count=0;
  int break_n=0;
  BasicBlock *cur_bb=NULL;
  map_n_b.clear();
  bb_vector.clear();
  bb_fakeins.clear();
  visit.clear();
  cut_pair.clear();
  int source_basic_num=0;
  for (bs=F.begin(), be=F.end(); bs!=be; bs++)
  {
    BasicBlock * test=&*bs;
    bb_vector.push_back(&*bs);
    map_n_b[test->getName().data()]=&*bs;
    source_basic_num++;
  }


  map_b.clear();
  map_b_r.clear();

  std::ifstream in_f;
  in_f.open(" ./passmatch.txt");//
  if( ! in_f.is_open() ) {
    errs() << "[ERROR]fail to open match.txt" << "\n";
    exit(0);
  }
  
  string match_l;
  match_l.clear();
  while(getline(in_f, match_l))//
  {
    string source, target;
    int pos=match_l.find(" ");
    source=match_l.substr(0,pos);
    target=match_l.substr(pos+1);
    map_b[source]=target;
    map_b_r[target]=source;
  }
  in_f.close();


  //
  //
  std::string str_suc =funcBname;
  std::ifstream in_b_suc;
  string func_dir=" ./passfunc/";
  
  in_b_suc.open(func_dir+str_suc+ "_suc.txt");//
  string suc_l;
  suc_l.clear();
  suc_num.clear();
  
  int num_e=bb_vector.size();//
  getline(in_b_suc, suc_l);//
  int pos=suc_l.find(" ");
  suc_l=suc_l.substr(0, pos);
  int num_insert=stoi(suc_l)-source_basic_num;//
  for (int i=0; i< num_insert; i++)
  {
    BasicBlock *new_bb=BasicBlock::Create(Context, "jmp", &F);  
    //
    //
    IRBuilder<NoFolder> builder(new_bb);
    //
    Value *x = ConstantInt::get(Type::getInt32Ty(Context), 3);
    Value *y = ConstantInt::get(Type::getInt32Ty(Context), 2);
    Value *z = ConstantInt::get(Type::getInt32Ty(Context), 4);
    Value *tmp2 = builder.CreateBinOp(Instruction::Add, x, y, "tmp");
    

    //



    PHINode* phi_ins=dyn_cast <PHINode>(new_bb->begin());  
    bb_vector.push_back(new_bb);   
    bb_fakeins.push_back(new_bb);
  }

  //
  
  int count_tmp=0;//
  while(getline(in_b_suc, suc_l))
  {
    string source, target;
    int pos=suc_l.find(" ");
    source=suc_l.substr(0, pos);
    string tail=suc_l.substr(pos+1);
    pos=tail.find(" ");
    target=tail.substr(0, pos);
    suc_num[source]=stoi(target);
    int i=stoi(target);
    tail=tail.substr(pos+1);
    while(i--)//
    {
      pos=tail.find(" ");
      target=tail.substr(0, pos);
      //
      suc_str[source].push_back(target);
      tail=tail.substr(pos+1);
    }

    if (map_b_r.count(source) == 0)//
    {
      std::string str = bb_vector[num_e+count_tmp]->getName().data();
      map_b[str] = source;
      map_n_b[str] = bb_vector[num_e+count_tmp];
      count_tmp+=1;
      map_b_r[source] = str;
    }
  }
  
  //
  //
  //
  //
  //
  //
  //
    
  //
  //
  //

  //
  //
  //
  //
  //
  //


   
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //
  //


  //
  //



  in_b_suc.close();

  for (int i=0; i<bb_vector.size(); i++)
  {
      //
      //
      cur_bb=bb_vector[i];
      string target_b=map_b[cur_bb->getName()];//
      create_line(cur_bb, target_b, suc_num[target_b], F);
  }
  
  //
  int num_recover=0;//
  num_recover=cut_pair.size();
  for(int i=0; i<num_recover; i++)//
  {
    BasicBlock* start=cut_pair[i].first;//
    BasicBlock* end=cut_pair[i].second;//
    bool flag=false;
    flag=dfs(start, end);
  }
  
  
  

}
bool SkeletonPass::dfs(BasicBlock* start, BasicBlock* end){
  BasicBlock* tmp=start;
  int num_suc=0;//

  //
  if(!(tmp->getTerminator()))
  {
    return false;
  }
  else if (isa<BranchInst>(tmp->getTerminator()))
  {
    BranchInst*br = cast<BranchInst>(tmp->getTerminator());
    if (br->isUnconditional())//
    {
      num_suc=1;
    }
    else if(br->isConditional())
    {
      num_suc=2;
    }

    Value* True=ConstantInt::getTrue(Context);
    Value* fake=ConstantInt::getFalse(Context);   
    for(int j=0; j<num_suc; j++ )
    {
      
      BasicBlock* suc = br->getSuccessor(j);
      if(suc == end)//
      {
        //
        //
        auto iter = std::remove(bb_fakeins.begin(), bb_fakeins.end(), tmp);
        bb_fakeins.erase(iter, bb_fakeins.end());

        if(num_suc==2){
          if(br->getCondition()==True || br->getCondition()==fake)
          {
            if(j==0){
              br->setCondition (True);
            }
            else if(j==1){
              br->setCondition (fake);
            }
          }
        }
        return true;
      }
      else 
      {
        bool flag=false;
        if(visit[suc]==0)//
        {
          string b_name=suc->getName().data();
          if(b_name.find("jmp")==b_name.npos){//
            continue;
          }
          visit[suc]=1;
          flag=dfs(suc, end);
          visit[suc]=0;
          if(flag)
          {
            //
            //
            auto iter = std::remove(bb_fakeins.begin(), bb_fakeins.end(), tmp);
            bb_fakeins.erase(iter, bb_fakeins.end());

            if(num_suc==2){
              //
              if(br->getCondition()==True || br->getCondition()==fake)
              {
                if(j==0){
                  br->setCondition (True);
                }
                else if(j==1){
                  br->setCondition (fake);
                }
              }
            }
            return true;
          }
          //
        }

      }
    }
    return false;
    //
  }
  else if (isa<SwitchInst>(tmp->getTerminator()))
  {
    //
    return false;
  }
    
} 

void SkeletonPass::insert_fakeins(void){
  IRBuilder<> B(Context);
  int num=0;
  BasicBlock* tmp=NULL;
  std::ifstream in_f;
  std::string filepath = " ./pass";
  in_f.open(filepath + "flag_tar.txt", ios_base::in);
  string s;
  map<string, int> len_v;//
  len_v.clear();

  map<string, int> s_pos;//
  s_pos.clear();
  while (getline(in_f, s))
  {
      int pos=0;
      pos=s.find(" ");
      string name= s.substr(0, pos);
      string tmp = s.substr(pos+1);
      pos=tmp.find(" ");
      string pos__;
      pos__=tmp.substr(0, pos);
      s_pos[name]=stoi(pos__);
      string length= tmp.substr(pos+1);
      int len=stoi(length);
      len_v[name]=len;
  }


  std::ofstream out_f;
  out_f.open(filepath + "flag_ori.txt", std::ios::out | std::ios::trunc);
      
  for(int i=0; i<bb_fakeins.size(); i++)
  {
    tmp=bb_fakeins[i];
    string tem_myasm = tmp->getName().data();
    string target_bname;
    target_bname.clear();
    target_bname=map_b[tem_myasm];
    int num, pos_s_n;
    num=len_v[target_bname];//
    pos_s_n=s_pos[target_bname];
    for(int i=0; i<num; i++)//
    {
      string nop_="nop";
      StringRef myasm_start = llvm::StringRef("nop");
      string cons="~{dirflag},~{fpsr},~{flags}";
      llvm::InlineAsm *IA = llvm::InlineAsm::get(FunctionType::get(B.getVoidTy(), false), myasm_start, StringRef(cons), true);
      BasicBlock* cur_bb=tmp;
      llvm::BasicBlock::iterator it=cur_bb->begin();
      //

      B.SetInsertPoint(cur_bb, it);
      B.CreateCall(IA, {});
    }
    
    
    string tem_myasm_start;
    tem_myasm_start=tem_myasm+"_s:";
    
    //
    StringRef myasm_start = llvm::StringRef(tem_myasm_start);
    string cons="~{dirflag},~{fpsr},~{flags}";
    llvm::InlineAsm *IA = llvm::InlineAsm::get(FunctionType::get(B.getVoidTy(), false), myasm_start, StringRef(cons), true);
    BasicBlock* cur_bb=tmp;
    llvm::BasicBlock::iterator it=cur_bb->begin();
    //

    B.SetInsertPoint(cur_bb, it);
    B.CreateCall(IA, {});


    it=cur_bb->end();
    it--;//
    string tem_myasm_end;
    tem_myasm_end=tem_myasm+"_e:";
    StringRef myasm_end = llvm::StringRef(tem_myasm_end);
    IA = llvm::InlineAsm::get(FunctionType::get(B.getVoidTy(), false), myasm_end, StringRef(cons), true);
    B.SetInsertPoint(cur_bb, it);
    B.CreateCall(IA, {});
    
    out_f << tem_myasm_start << " " << tem_myasm_end << " " << pos_s_n << " " << num << "\n";//
    

    Value* True=ConstantInt::getTrue(Context);
    if (isa<BranchInst>(tmp->getTerminator()))
    {
      BranchInst*br=cast<BranchInst>(tmp->getTerminator());
      if (br->isConditional())//
      {
        br->setCondition (True);
        string t_s, s_t_;
        t_s= tmp->getName().data();
        s_t_="jmp1";
        if(True)//
        {
          BasicBlock* i1= br->getSuccessor(0);
          BasicBlock* i2= br->getSuccessor(1);
          br->setSuccessor(0, i2);
          br->setSuccessor(1, i1);
        }
        //
      }
    }
  }
  in_f.close();
  out_f.close();
  return;
}




char SkeletonPass::ID = 0;

//
//
//
//
//
//
//
//
//


//
static RegisterPass<SkeletonPass> X("hello", "Hello World Pass",
                          false /* Only looks at CFG */,
                          false /* Analysis Pass */);
//
//
//
//         llvm::legacy::PassManagerBase &PM) { PM.add(new Hello()); });