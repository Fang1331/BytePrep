import React, { useState, useRef, useEffect } from 'react';
import './App.css';

// Monaco Editor Component
const CodeEditor = ({ value, onChange, language, height = "400px" }) => {
  const editorRef = useRef(null);
  const monacoRef = useRef(null);

  useEffect(() => {
    // Load Monaco Editor dynamically
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/monaco-editor@0.45.0/min/vs/loader.js';
    script.onload = () => {
      window.require.config({ paths: { vs: 'https://cdn.jsdelivr.net/npm/monaco-editor@0.45.0/min/vs' }});
      window.require(['vs/editor/editor.main'], () => {
        if (editorRef.current && !monacoRef.current) {
          monacoRef.current = window.monaco.editor.create(editorRef.current, {
            value: value || '',
            language: language || 'python',
            theme: 'vs-dark',
            fontSize: 14,
            lineNumbers: 'on',
            roundedSelection: false,
            scrollBeyondLastLine: false,
            readOnly: false,
            automaticLayout: true,
            minimap: { enabled: false },
            scrollbar: {
              useShadows: false,
              verticalHasArrows: true,
              horizontalHasArrows: true,
              vertical: 'auto',
              horizontal: 'auto'
            }
          });

          monacoRef.current.onDidChangeModelContent(() => {
            const currentValue = monacoRef.current.getValue();
            if (onChange) onChange(currentValue);
          });
        }
      });
    };
    document.head.appendChild(script);

    return () => {
      if (monacoRef.current) {
        monacoRef.current.dispose();
        monacoRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (monacoRef.current && value !== monacoRef.current.getValue()) {
      monacoRef.current.setValue(value || '');
    }
  }, [value]);

  useEffect(() => {
    if (monacoRef.current && language) {
      const model = monacoRef.current.getModel();
      window.monaco.editor.setModelLanguage(model, language);
    }
  }, [language]);

  return <div ref={editorRef} style={{ height, width: '100%' }} />;
};

// Problem Card Component
const ProblemCard = ({ problem, onSelect, isSelected }) => (
  <div 
    className={`p-4 border rounded-lg cursor-pointer transition-all ${
      isSelected ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-gray-300'
    }`}
    onClick={() => onSelect(problem)}
  >
    <div className="flex justify-between items-start mb-2">
      <h3 className="font-semibold text-lg">{problem.title}</h3>
      <span className={`px-2 py-1 text-xs rounded-full ${
        problem.difficulty === 'easy' ? 'bg-green-100 text-green-800' :
        problem.difficulty === 'medium' ? 'bg-yellow-100 text-yellow-800' :
        'bg-red-100 text-red-800'
      }`}>
        {problem.difficulty}
      </span>
    </div>
    <p className="text-gray-600 text-sm mb-2">{problem.description.slice(0, 150)}...</p>
    <span className="inline-block bg-gray-100 text-gray-700 px-2 py-1 text-xs rounded">
      {problem.category}
    </span>
  </div>
);

// Feedback Component
const FeedbackPanel = ({ evaluation, isLoading }) => {
  if (isLoading) {
    return (
      <div className="p-6 bg-white rounded-lg shadow-sm border">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-300 rounded w-1/4 mb-4"></div>
          <div className="space-y-3">
            <div className="h-3 bg-gray-300 rounded"></div>
            <div className="h-3 bg-gray-300 rounded w-5/6"></div>
            <div className="h-3 bg-gray-300 rounded w-4/6"></div>
          </div>
        </div>
      </div>
    );
  }

  if (!evaluation) return null;

  return (
    <div className="space-y-6">
      {/* Overall Score */}
      <div className="p-6 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg border">
        <h3 className="text-lg font-semibold mb-4">Overall Assessment</h3>
        <div className="flex items-center justify-between">
          <div>
            <div className="text-3xl font-bold text-blue-600">
              {Math.round(evaluation.overall_score)}/100
            </div>
            <div className="text-sm text-gray-600">Overall Score</div>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-600">Execution Time</div>
            <div className="font-semibold">{evaluation.execution_time?.toFixed(2)}s</div>
          </div>
        </div>
      </div>

      {/* Dual AI Feedback */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* StarCoder Analysis */}
        <div className="p-6 bg-white rounded-lg shadow-sm border">
          <h4 className="font-semibold text-green-700 mb-3 flex items-center">
            <span className="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
            Performance Analysis
          </h4>
          <div className="space-y-3">
            <div className="text-sm">
              <span className="font-medium">Efficiency Score:</span> 
              <span className="ml-2 font-semibold text-green-600">
                {evaluation.starcoder_feedback?.efficiency_score || 'N/A'}/100
              </span>
            </div>
            <div className="text-sm">
              <span className="font-medium">Time Complexity:</span> 
              <span className="ml-2">{evaluation.starcoder_feedback?.time_complexity || 'N/A'}</span>
            </div>
            <div className="text-sm">
              <span className="font-medium">Space Complexity:</span> 
              <span className="ml-2">{evaluation.starcoder_feedback?.space_complexity || 'N/A'}</span>
            </div>
          </div>
        </div>

        {/* Gemini Analysis */}
        <div className="p-6 bg-white rounded-lg shadow-sm border">
          <h4 className="font-semibold text-purple-700 mb-3 flex items-center">
            <span className="w-2 h-2 bg-purple-500 rounded-full mr-2"></span>
            Interview Readiness
          </h4>
          <div className="space-y-3">
            <div className="text-sm">
              <span className="font-medium">Correctness:</span> 
              <span className="ml-2 font-semibold text-purple-600">
                {evaluation.gemini_feedback?.correctness_score || 'N/A'}/100
              </span>
            </div>
            <div className="text-sm">
              <span className="font-medium">Readability:</span> 
              <span className="ml-2 font-semibold text-purple-600">
                {evaluation.gemini_feedback?.readability_score || 'N/A'}/100
              </span>
            </div>
            <div className="text-sm">
              <span className="font-medium">Interview Score:</span> 
              <span className="ml-2 font-semibold text-purple-600">
                {evaluation.gemini_feedback?.interview_score || 'N/A'}/100
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Detailed Feedback */}
      {evaluation.starcoder_feedback?.analysis && (
        <div className="p-6 bg-white rounded-lg shadow-sm border">
          <h4 className="font-semibold mb-3">Performance Analysis Details</h4>
          <div className="prose prose-sm max-w-none">
            <p className="text-gray-700 whitespace-pre-wrap">{evaluation.starcoder_feedback.analysis}</p>
          </div>
        </div>
      )}

      {evaluation.gemini_feedback?.feedback && (
        <div className="p-6 bg-white rounded-lg shadow-sm border">
          <h4 className="font-semibold mb-3">Interview Feedback</h4>
          <div className="prose prose-sm max-w-none">
            <p className="text-gray-700 whitespace-pre-wrap">{evaluation.gemini_feedback.feedback}</p>
          </div>
        </div>
      )}

      {/* Suggestions */}
      {evaluation.suggestions?.length > 0 && (
        <div className="p-6 bg-white rounded-lg shadow-sm border">
          <h4 className="font-semibold mb-3">Improvement Suggestions</h4>
          <ul className="space-y-2">
            {evaluation.suggestions.map((suggestion, index) => (
              <li key={index} className="flex items-start">
                <span className="w-1.5 h-1.5 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                <span className="text-gray-700">{suggestion}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

// Main App Component
function App() {
  const [problems, setProblems] = useState([]);
  const [selectedProblem, setSelectedProblem] = useState(null);
  const [code, setCode] = useState('');
  const [language, setLanguage] = useState('python');
  const [evaluation, setEvaluation] = useState(null);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [activeTab, setActiveTab] = useState('problems');

  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

  // Fetch problems on component mount
  useEffect(() => {
    fetchProblems();
  }, []);

  const fetchProblems = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/problems`);
      const data = await response.json();
      setProblems(data.problems || []);
    } catch (error) {
      console.error('Failed to fetch problems:', error);
    }
  };

  const handleProblemSelect = (problem) => {
    setSelectedProblem(problem);
    setActiveTab('editor');
    setEvaluation(null);
    
    // Set starter code based on language
    const starterCode = {
      python: `def solution():\n    # Your solution here\n    pass`,
      javascript: `function solution() {\n    // Your solution here\n}`,
      java: `public class Solution {\n    public void solution() {\n        // Your solution here\n    }\n}`,
      cpp: `#include <iostream>\nusing namespace std;\n\nclass Solution {\npublic:\n    void solution() {\n        // Your solution here\n    }\n};`,
    };
    
    setCode(starterCode[language] || starterCode.python);
  };

  const handleEvaluateCode = async () => {
    if (!selectedProblem || !code.trim()) {
      alert('Please select a problem and write some code');
      return;
    }

    setIsEvaluating(true);
    setEvaluation(null);

    try {
      const response = await fetch(`${BACKEND_URL}/api/evaluate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          code: code,
          language: language,
          problem_description: selectedProblem.description,
          difficulty: selectedProblem.difficulty
        }),
      });

      if (!response.ok) {
        throw new Error('Evaluation failed');
      }

      const result = await response.json();
      setEvaluation(result);
      setActiveTab('feedback');
    } catch (error) {
      console.error('Evaluation failed:', error);
      alert('Evaluation failed. Please try again.');
    } finally {
      setIsEvaluating(false);
    }
  };

  const languageOptions = [
    { value: 'python', label: 'Python' },
    { value: 'javascript', label: 'JavaScript' },
    { value: 'java', label: 'Java' },
    { value: 'cpp', label: 'C++' },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-900">BytePrep</h1>
              <span className="ml-2 px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                AI-Powered
              </span>
            </div>
            <nav className="flex space-x-1">
              <button
                onClick={() => setActiveTab('problems')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'problems' 
                    ? 'bg-blue-100 text-blue-700' 
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                Problems
              </button>
              <button
                onClick={() => setActiveTab('editor')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'editor' 
                    ? 'bg-blue-100 text-blue-700' 
                    : 'text-gray-500 hover:text-gray-700'
                }`}
                disabled={!selectedProblem}
              >
                Code Editor
              </button>
              <button
                onClick={() => setActiveTab('feedback')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'feedback' 
                    ? 'bg-blue-100 text-blue-700' 
                    : 'text-gray-500 hover:text-gray-700'
                }`}
                disabled={!evaluation && !isEvaluating}
              >
                AI Feedback
              </button>
            </nav>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Problems Tab */}
        {activeTab === 'problems' && (
          <div>
            <div className="mb-6">
              <h2 className="text-xl font-semibold mb-2">Coding Problems</h2>
              <p className="text-gray-600">Choose a problem to start practicing with AI-powered feedback</p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {problems.map((problem) => (
                <ProblemCard
                  key={problem.id}
                  problem={problem}
                  onSelect={handleProblemSelect}
                  isSelected={selectedProblem?.id === problem.id}
                />
              ))}
            </div>
          </div>
        )}

        {/* Code Editor Tab */}
        {activeTab === 'editor' && (
          <div>
            {selectedProblem ? (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Problem Description */}
                <div className="space-y-4">
                  <div className="bg-white p-6 rounded-lg shadow-sm border">
                    <div className="flex justify-between items-start mb-4">
                      <h2 className="text-xl font-semibold">{selectedProblem.title}</h2>
                      <span className={`px-3 py-1 text-sm rounded-full ${
                        selectedProblem.difficulty === 'easy' ? 'bg-green-100 text-green-800' :
                        selectedProblem.difficulty === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-red-100 text-red-800'
                      }`}>
                        {selectedProblem.difficulty}
                      </span>
                    </div>
                    
                    <p className="text-gray-700 mb-4">{selectedProblem.description}</p>
                    
                    {selectedProblem.examples && selectedProblem.examples.length > 0 && (
                      <div>
                        <h4 className="font-medium mb-2">Examples:</h4>
                        {selectedProblem.examples.map((example, index) => (
                          <div key={index} className="bg-gray-50 p-3 rounded mb-2">
                            <div className="text-sm">
                              <div><strong>Input:</strong> {example.input}</div>
                              <div><strong>Output:</strong> {example.output}</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                {/* Code Editor */}
                <div className="space-y-4">
                  <div className="bg-white rounded-lg shadow-sm border overflow-hidden">
                    <div className="p-4 bg-gray-50 border-b flex justify-between items-center">
                      <div className="flex items-center space-x-4">
                        <select
                          value={language}
                          onChange={(e) => setLanguage(e.target.value)}
                          className="border rounded px-3 py-1 text-sm"
                        >
                          {languageOptions.map(lang => (
                            <option key={lang.value} value={lang.value}>
                              {lang.label}
                            </option>
                          ))}
                        </select>
                      </div>
                      <button
                        onClick={handleEvaluateCode}
                        disabled={isEvaluating || !code.trim()}
                        className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
                      >
                        {isEvaluating ? (
                          <>
                            <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            Analyzing...
                          </>
                        ) : (
                          'Get AI Feedback'
                        )}
                      </button>
                    </div>
                    
                    <CodeEditor
                      value={code}
                      onChange={setCode}
                      language={language}
                      height="500px"
                    />
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <div className="max-w-md mx-auto">
                  <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                  </svg>
                  <h3 className="mt-2 text-sm font-medium text-gray-900">No problem selected</h3>
                  <p className="mt-1 text-sm text-gray-500">Choose a problem from the Problems tab to start coding.</p>
                  <div className="mt-6">
                    <button
                      onClick={() => setActiveTab('problems')}
                      className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700"
                    >
                      View Problems
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Feedback Tab */}
        {activeTab === 'feedback' && (
          <div>
            <div className="mb-6">
              <h2 className="text-xl font-semibold mb-2">AI-Powered Feedback</h2>
              <p className="text-gray-600">Dual AI analysis from StarCoder2 and Gemini Pro</p>
            </div>
            
            <FeedbackPanel evaluation={evaluation} isLoading={isEvaluating} />
            
            {!evaluation && !isEvaluating && (
              <div className="text-center py-12">
                <div className="max-w-md mx-auto">
                  <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                  <h3 className="mt-2 text-sm font-medium text-gray-900">No evaluation yet</h3>
                  <p className="mt-1 text-sm text-gray-500">Submit your code in the Editor tab to get AI feedback.</p>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;