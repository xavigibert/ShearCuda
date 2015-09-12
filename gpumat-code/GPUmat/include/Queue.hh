/*
     Copyright (C) 2012  GP-you Group (http://gp-you.org)
 
     This program is free software: you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation, either version 3 of the License, or
     (at your option) any later version.
 
     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.
 
     You should have received a copy of the GNU General Public License
     along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

template<class C>
class Queue {
private:

  class ListNode {
  public:
    C* element;
    ListNode *next;

    ListNode(C *theElement, ListNode * n = NULL) :
    element(theElement), next(n) {
    }
  };

  ListNode *front;
  ListNode *back;

public:
  Queue<C>() :
      front(NULL), back(NULL) {
      }

      ~Queue<C>() {
        makeEmpty();
      }

      void makeEmpty() {
        while (!isEmpty())
          dequeue();
      }

      bool isEmpty() const {
        return front == NULL;
      }

      C* getFront() const {
        if (isEmpty())
          return NULL;
        else
          return front->element;
      }

      void enqueue(C *x) {
        if (isEmpty())
          back = front = new ListNode(x);
        else
          back = back->next = new ListNode(x);
      }

      C* dequeue() {
        C* frontItem = getFront();
        ListNode *old = front;
        front = front->next;
        delete old;
        return frontItem;
      }

      C* getElementAt(int n) {
        ListNode *start = front;
        // start counting
        ListNode * ret = NULL;
        for (int i=0;i<=n;i++) {
          if (start==NULL)
            return NULL;
          ret = start;
          start = start->next;
        }
        return ret->element;
      }

};
